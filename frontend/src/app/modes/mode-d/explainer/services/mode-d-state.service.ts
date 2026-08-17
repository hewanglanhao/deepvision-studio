import { Injectable, computed, inject, signal } from '@angular/core';
import {
  ModeDAttentionSummary,
  ModeDExample,
  ModeDInferenceResult,
  ModeDQkvTeachingData,
  ModeDReportSection,
  ModeDTokenScore,
  ModeDVectorBar
} from '../models/mode-d.types';
import { ModeDAssetsService } from './mode-d-assets.service';
import { ModeDInferenceService } from './mode-d-inference.service';

@Injectable({ providedIn: 'root' })
export class ModeDStateService {
  private readonly assets = inject(ModeDAssetsService);
  private readonly inference = inject(ModeDInferenceService);

  readonly examples = this.assets.examples;
  readonly blockOptions = this.assets.blockOptions;
  readonly headOptions = this.assets.headOptions;

  readonly selectedExampleId = signal(this.examples[0]?.id ?? '');
  readonly inputText = signal(this.examples[0]?.text ?? '');
  readonly selectedBlockIndex = signal(0);
  readonly selectedHeadIndex = signal(0);
  readonly hoveredCell = signal<{ row: number; col: number } | null>(null);
  readonly selectedCell = signal<{ row: number; col: number } | null>(null);
  readonly inferenceLoading = signal(false);
  readonly inferenceError = signal('');
  readonly inferenceResult = signal<ModeDInferenceResult | null>(null);
  readonly inferenceSource = signal<'loading' | 'real' | 'fallback'>('loading');

  readonly inferenceSourceLabel = computed(() => {
    switch (this.inferenceSource()) {
      case 'real': return '真实 GPT-2 ONNX 推理';
      case 'fallback': return '教学降级数据';
      default: return '正在加载模型资源';
    }
  });

  readonly currentExample = computed<ModeDExample | null>(() =>
    this.examples.find(example => example.id === this.selectedExampleId()) ?? null
  );

  readonly tokens = computed(() => this.inferenceResult()?.tokenTexts ?? this.fallbackTokens());
  readonly tokenIds = computed(() => this.inferenceResult()?.tokenIds ?? this.fallbackTokenIds());
  readonly topK = computed<ModeDTokenScore[]>(() => this.inferenceResult()?.topK ?? this.buildFallbackTopK());

  readonly attentionMatrix = computed<number[][]>(() => {
    const result = this.inferenceResult();
    const key = this.getAttentionKey();
    const matrix = key ? result?.attentionByKey[key] : undefined;
    if (matrix && matrix.length) {
      return matrix;
    }

    const tokens = this.tokens();
    return tokens.map((_, row) => {
      const raw = tokens.map((__, col) => {
        const distance = Math.abs(row - col);
        const recency = row >= col ? 1 / (distance + 1.5) : 0.05;
        return recency + (col === row ? 0.14 : 0);
      });
      const total = raw.reduce((sum, value) => sum + value, 0);
      return raw.map(value => value / total);
    });
  });

  readonly strongestAttention = computed<ModeDAttentionSummary>(() => {
    const matrix = this.attentionMatrix();
    const tokens = this.tokens();
    const strongest = this.findStrongestCell(matrix) ?? { row: 0, col: 0 };
    const sourceToken = tokens[strongest.row] ?? '';
    const targetToken = tokens[strongest.col] ?? '';
    const narrative = this.selectedHeadIndex() === 0
      ? '当前这个头更偏向最近上下文，适合解释局部续写线索。'
      : '当前这个头更偏向句首或结构锚点，适合解释全局依赖与主题回看。';

    return {
      sourceToken,
      targetToken,
      weight: matrix[strongest.row]?.[strongest.col] ?? 0,
      narrative
    };
  });

  readonly activeAttentionDetail = computed<{
    row: number;
    col: number;
    weight: number;
    sourceToken: string;
    targetToken: string;
    interpretation: string;
    mode: 'selected' | 'hovered' | 'strongest';
  }>(() => {
    const matrix = this.attentionMatrix();
    const tokens = this.tokens();
    const selected = this.selectedCell();
    const hovered = this.hoveredCell();
    const fallback = this.findStrongestCell(matrix);
    const current = selected ?? hovered ?? fallback ?? { row: 0, col: 0 };

    const row = current.row;
    const col = current.col;
    const weight = matrix[row]?.[col] ?? 0;
    const sourceToken = tokens[row] ?? '';
    const targetToken = tokens[col] ?? '';

    let interpretation = '这一格表示当前 token 对另一个 token 的注意力分配。';
    if (row === col) {
      interpretation = '这是自注意力位置，说明当前 token 会保留自身信息。';
    } else if (col === Math.max(0, row - 1)) {
      interpretation = '这一格通常对应最近上下文依赖，适合解释语言建模中的局部续写。';
    } else if (col === 0) {
      interpretation = '这一格说明当前头会回看序列开头，常用来捕捉句首结构或全局主题。';
    } else if (col < row) {
      interpretation = '这一格说明模型正在利用更早出现的上下文来决定当前 token 的表示。';
    }

    return {
      row,
      col,
      weight,
      sourceToken,
      targetToken,
      interpretation,
      mode: selected ? 'selected' : hovered ? 'hovered' : 'strongest'
    };
  });

  readonly qkvTeaching = computed<ModeDQkvTeachingData>(() => {
    const focus = this.activeAttentionDetail();
    const tokenIds = this.tokenIds();
    const queryId = tokenIds[focus.row] ?? focus.row + 1;
    const keyId = tokenIds[focus.col] ?? focus.col + 1;
    const attentionWeight = focus.weight;

    const queryVector = this.createTeachingVector(queryId, 0.42);
    const keyVector = this.createTeachingVector(keyId, 0.34);
    const valueVector = this.createTeachingVector(keyId + 7, 0.27);

    const summary = [
      `在教学视角下，“${focus.sourceToken}”被看作 Query，它携带“当前想找什么”的检索意图。`,
      `“${focus.targetToken}”被看作 Key 和 Value：Key 用来回答“我是否相关”，Value 负责在匹配后把语义内容传回输出。`,
      `当前这条注意力连接的权重约为 ${(attentionWeight * 100).toFixed(1)}%，说明这个头会把一部分来自“${focus.targetToken}”的信息传回给“${focus.sourceToken}”。`
    ].join('');

    return {
      queryToken: focus.sourceToken,
      keyToken: focus.targetToken,
      valueToken: focus.targetToken,
      queryIndex: focus.row,
      keyIndex: focus.col,
      attentionWeight,
      queryVector,
      keyVector,
      valueVector,
      summary
    };
  });

  readonly reportSections = computed<ModeDReportSection[]>(() => {
    const example = this.currentExample();
    const topK = this.topK();
    const strongest = this.strongestAttention();
    const focus = this.activeAttentionDetail();
    const qkv = this.qkvTeaching();
    const summary = this.generatedExplanation();

    return [
      {
        title: '推理数据源',
        body: this.inferenceSource() === 'real'
          ? '当前 Top-K、token 和注意力矩阵来自浏览器内运行的 GPT-2 ONNX 模型。'
          : '当前使用教学降级数据，不代表 GPT-2 的真实模型输出。请先安装 Mode D 资源后重新推理。'
      },
      {
        title: '当前样例',
        body: example
          ? `${example.title}：${example.subtitle}。当前输入为“${this.inputText()}”。`
          : `当前输入为“${this.inputText()}”。`
      },
      {
        title: 'Top-5 预测',
        body: topK
          .slice(0, 5)
          .map(item => `${item.rank}. ${item.token} ${(item.probability * 100).toFixed(1)}%`)
          .join('；')
      },
      {
        title: '注意力观察',
        body: `在 ${this.blockOptions[this.selectedBlockIndex()]?.label ?? '当前层'} 的 ${this.headOptions[this.selectedHeadIndex()]?.label ?? '当前头'} 中，token “${strongest.sourceToken}” 对 “${strongest.targetToken}” 的注意力最高，权重约 ${(strongest.weight * 100).toFixed(1)}%。`
      },
      {
        title: '当前聚焦单元',
        body: `当前聚焦在第 ${focus.row + 1} 行、第 ${focus.col + 1} 列，对应 “${focus.sourceToken}” -> “${focus.targetToken}”，权重 ${(focus.weight * 100).toFixed(1)}%。${focus.interpretation}`
      },
      {
        title: 'QKV 教学解释',
        body: qkv.summary
      },
      {
        title: '自动解释',
        body: summary
      }
    ];
  });

  constructor() {
    queueMicrotask(() => {
      void this.runInference();
    });
  }

  applyExample(exampleId: string): void {
    const example = this.examples.find(item => item.id === exampleId);
    if (!example) return;
    this.selectedExampleId.set(example.id);
    this.inputText.set(example.text);
    void this.runInference();
  }

  updateInputText(value: string): void {
    this.inputText.set(value);
  }

  selectBlock(index: number): void {
    this.selectedBlockIndex.set(index);
    this.selectedCell.set(null);
    this.hoveredCell.set(null);
  }

  selectHead(index: number): void {
    this.selectedHeadIndex.set(index);
    this.selectedCell.set(null);
    this.hoveredCell.set(null);
  }

  hoverAttentionCell(row: number, col: number): void {
    this.hoveredCell.set({ row, col });
  }

  clearHoveredAttentionCell(): void {
    this.hoveredCell.set(null);
  }

  selectAttentionCell(row: number, col: number): void {
    const current = this.selectedCell();
    if (current?.row === row && current?.col === col) {
      this.selectedCell.set(null);
      return;
    }
    this.selectedCell.set({ row, col });
  }

  generatedExplanation(): string {
    const example = this.currentExample();
    const top1 = this.topK()[0];
    const strongest = this.strongestAttention();
    const focus = this.activeAttentionDetail();
    const qkv = this.qkvTeaching();
    const block = this.blockOptions[this.selectedBlockIndex()]?.label ?? '当前层';
    const head = this.headOptions[this.selectedHeadIndex()]?.label ?? '当前头';

    return [
      `当前数据源：${this.inferenceSourceLabel()}。`,
      '当前页面重点解释下一词预测和单头注意力如何共同决定模型输出。',
      `当前输入末尾语境让模型最倾向输出“${top1?.token ?? ''}”，概率约 ${(((top1?.probability) ?? 0) * 100).toFixed(1)}%。`,
      `${block} 的 ${head} 主要把注意力从“${strongest.sourceToken}”指向“${strongest.targetToken}”，说明模型正在利用这部分上下文决定下一词分布。`,
      `当前聚焦单元展示的是“${focus.sourceToken}”如何关注“${focus.targetToken}”，其权重约 ${(focus.weight * 100).toFixed(1)}%。${focus.interpretation}`,
      qkv.summary,
      example?.focus ?? ''
    ].join('');
  }

  async runInference(): Promise<void> {
    this.inferenceLoading.set(true);
    this.inferenceError.set('');
    this.inferenceSource.set('loading');
    try {
      const result = await this.inference.runInference(this.inputText(), 10);
      this.inferenceResult.set(result);
      this.inferenceSource.set('real');
      this.hoveredCell.set(null);
      this.selectedCell.set(null);
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Transformer 推理失败。';
      this.inferenceResult.set(null);
      this.inferenceSource.set('fallback');
      this.inferenceError.set(`${message} 请在 frontend 目录执行 npm run setup:mode-d 后重试。`);
    } finally {
      this.inferenceLoading.set(false);
    }
  }

  private fallbackTokens(): string[] {
    const normalized = this.inputText().trim().replace(/\s+/g, ' ');
    return normalized ? normalized.split(' ').slice(0, 12) : ['<blank>'];
  }

  private fallbackTokenIds(): number[] {
    return this.fallbackTokens().map((token, index) => {
      const seed = Array.from(token).reduce((sum, char) => sum + char.charCodeAt(0), 0);
      return seed + index;
    });
  }

  private buildFallbackTopK(): ModeDTokenScore[] {
    const candidates = this.currentExample()?.candidateTokens ?? ['token', 'context', 'head', 'attention', 'model'];
    const rawScores = candidates.slice(0, 5).map((_, index) => 1 / (index + 2));
    const total = rawScores.reduce((sum, value) => sum + value, 0);

    return candidates.slice(0, 5).map((token, index) => ({
      tokenId: index,
      token,
      probability: rawScores[index]! / total,
      rank: index + 1
    }));
  }

  private getAttentionKey(): string | null {
    const block = this.selectedBlockIndex();
    const head = this.selectedHeadIndex();
    return `block_${block}_attn_head_${head}_attn_dropout`;
  }

  private findStrongestCell(matrix: number[][]): { row: number; col: number } | null {
    let best = { row: 0, col: 0, value: -Infinity };

    matrix.forEach((rowValues, row) => {
      rowValues.forEach((value, col) => {
        if (value > best.value) {
          best = { row, col, value };
        }
      });
    });

    return Number.isFinite(best.value) ? { row: best.row, col: best.col } : null;
  }

  private createTeachingVector(seed: number, amplitude: number): ModeDVectorBar[] {
    return Array.from({ length: 6 }, (_, index) => {
      const angle = seed * 0.17 + index * 0.81;
      const wave = Math.sin(angle) * amplitude + Math.cos(angle * 0.63) * 0.12;
      const value = Math.max(-1, Math.min(1, wave));
      return {
        label: `d${index + 1}`,
        value: Number(value.toFixed(3))
      };
    });
  }
}
