import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';
import {
  InferenceSampleItem,
  SingleInferenceActivation,
  SingleInferenceResult,
  TrainingCheckpointSummary,
  TrainingRuntimeService
} from '@shared/training/training-runtime.service';
import { LlmFloatingAssistantComponent, LlmQuickPrompt } from '@shared/llm/llm-floating-assistant.component';
import { LlmChatContext } from '@shared/llm/llm.models';
import { MODE_B_LLM_SYSTEM_PROMPT } from '@shared/llm/llm-prompts';
import { TeachingSearchFabComponent } from '@shared/teaching/teaching-search-fab.component';
import { TeachingTermDirective } from '@shared/teaching/teaching-term.directive';

@Component({
  selector: 'app-single-inference-page',
  standalone: true,
  imports: [
    CommonModule,
    RouterLink,
    PlatformTopbarComponent,
    LlmFloatingAssistantComponent,
    TeachingSearchFabComponent,
    TeachingTermDirective
  ],
  templateUrl: './single-inference-page.component.html',
  styleUrl: './single-inference-page.component.css'
})
export class SingleInferencePageComponent implements OnInit, OnDestroy {
  readonly Math = Math;
  authUser: AuthUser | null = null;
  checkpoints: TrainingCheckpointSummary[] = [];
  selectedCheckpointId: number | null = null;
  samples: InferenceSampleItem[] = [];
  selectedSample: InferenceSampleItem | null = null;
  inferenceResult: SingleInferenceResult | null = null;
  activeActivationOrder = 0;
  sampleDialogOpen = false;
  rawSampleDetail: InferenceSampleItem | null = null;
  hoveredCheckpointId: number | null = null;
  checkpointDetailPosition: { left: number; top: number } | null = null;
  loadingCheckpoints = false;
  loadingSamples = false;
  inferring = false;
  error = '';
  private readonly subs = new Subscription();
  private playTimer: number | null = null;

  readonly topbarStatusPills = ['单样本推理', '逐层激活', 'Checkpoint'];
  readonly inferenceLlmSystemPrompt = [
    MODE_B_LLM_SYSTEM_PROMPT,
    '',
    '当前页面是单样本推理页。请结合 checkpoint、原始样本、预测结果和逐层激活解释模型为何得到该结果，区分确定信息与推测，并指出置信度、激活或数据方面的异常。'
  ].join('\n');
  readonly inferenceLlmContextProvider = (): LlmChatContext => this.buildInferenceLlmContext();
  readonly inferenceLlmQuickPrompts: LlmQuickPrompt[] = [
    { label: '解释预测', question: '请解释当前样本为什么得到这个预测结果。' },
    { label: '分析激活', question: '请分析逐层激活变化，指出哪些层最值得关注。' },
    { label: '判断可信度', question: '请判断当前预测是否可信，并说明依据和风险。' },
    { label: '排查错误', question: '如果当前预测错误，请分析可能原因和改进方向。' }
  ];

  constructor(
    private authSvc: AuthService,
    private trainingSvc: TrainingRuntimeService
  ) {}

  ngOnInit(): void {
    this.subs.add(this.authSvc.user$.subscribe(user => {
      this.authUser = user;
      if (user) {
        void this.loadCheckpoints();
      } else {
        this.checkpoints = [];
        this.selectedCheckpointId = null;
        this.samples = [];
        this.selectedSample = null;
        this.inferenceResult = null;
      }
    }));
    void this.authSvc.restoreSession();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    this.stopActivationPlayback();
  }

  get completedCheckpoints(): TrainingCheckpointSummary[] {
    return this.checkpoints.filter(checkpoint => this.isCompletedCheckpoint(checkpoint));
  }

  get selectedCheckpoint(): TrainingCheckpointSummary | null {
    return this.completedCheckpoints.find(checkpoint => checkpoint.id === this.selectedCheckpointId) ?? this.completedCheckpoints[0] ?? null;
  }

  get checkpointDetail(): TrainingCheckpointSummary | null {
    return this.completedCheckpoints.find(checkpoint => checkpoint.id === this.hoveredCheckpointId) ?? null;
  }

  get activeActivation(): SingleInferenceActivation | null {
    const activations = this.inferenceResult?.activations ?? [];
    return activations.find(item => item.order === this.activeActivationOrder) ?? activations[0] ?? null;
  }

  get samplePreviewKind(): 'image' | 'table' {
    return this.samples.some(sample => !!sample.imageUrl) ? 'image' : 'table';
  }

  async loadCheckpoints(): Promise<void> {
    if (!this.authUser) return;
    this.loadingCheckpoints = true;
    this.error = '';
    try {
      this.checkpoints = await this.trainingSvc.listCheckpoints();
      if (!this.selectedCheckpointId || !this.completedCheckpoints.some(item => item.id === this.selectedCheckpointId)) {
        this.selectedCheckpointId = this.completedCheckpoints[0]?.id ?? null;
      }
      this.hideCheckpointPreview();
      this.samples = [];
      this.selectedSample = null;
      this.inferenceResult = null;
      this.rawSampleDetail = null;
    } catch (err) {
      this.error = err instanceof Error ? err.message : '加载 checkpoint 失败。';
    } finally {
      this.loadingCheckpoints = false;
    }
  }

  onCheckpointChange(): void {
    this.samples = [];
    this.selectedSample = null;
    this.inferenceResult = null;
    this.rawSampleDetail = null;
    this.stopActivationPlayback();
  }

  selectCheckpoint(checkpointId: number): void {
    if (this.selectedCheckpointId === checkpointId) return;
    this.selectedCheckpointId = checkpointId;
    this.onCheckpointChange();
  }

  previewCheckpoint(checkpointId: number, event?: Event): void {
    this.hoveredCheckpointId = checkpointId;
    this.placeCheckpointDetail(event?.currentTarget ?? null);
  }

  hideCheckpointPreview(): void {
    this.hoveredCheckpointId = null;
    this.checkpointDetailPosition = null;
  }

  async openSampleDialog(): Promise<void> {
    const checkpoint = this.selectedCheckpoint;
    if (!checkpoint) return;
    this.sampleDialogOpen = true;
    if (this.samples.length) return;
    this.loadingSamples = true;
    this.error = '';
    try {
      const result = await this.trainingSvc.listInferenceSamples(checkpoint.id, 72);
      this.samples = result.samples ?? [];
    } catch (err) {
      this.error = err instanceof Error ? err.message : '加载样本失败。';
    } finally {
      this.loadingSamples = false;
    }
  }

  chooseSample(sample: InferenceSampleItem): void {
    this.selectedSample = sample;
    this.sampleDialogOpen = false;
    this.rawSampleDetail = null;
    this.inferenceResult = null;
    this.stopActivationPlayback();
  }

  openRawSampleDetail(sample: InferenceSampleItem, event?: MouseEvent): void {
    event?.stopPropagation();
    this.rawSampleDetail = sample;
  }

  closeRawSampleDetail(): void {
    this.rawSampleDetail = null;
  }

  async runInference(): Promise<void> {
    const checkpoint = this.selectedCheckpoint;
    const sample = this.selectedSample;
    if (!checkpoint || !sample) return;
    this.inferring = true;
    this.error = '';
    this.stopActivationPlayback();
    try {
      this.inferenceResult = await this.trainingSvc.inferCheckpointSample(checkpoint.id, sample.index);
      this.selectedSample = this.inferenceResult.sample ?? sample;
      this.activeActivationOrder = 0;
      this.startActivationPlayback();
    } catch (err) {
      this.error = err instanceof Error ? err.message : '单样本推理失败。';
    } finally {
      this.inferring = false;
    }
  }

  selectActivation(order: number): void {
    this.activeActivationOrder = order;
    this.stopActivationPlayback();
  }

  startActivationPlayback(): void {
    this.stopActivationPlayback();
    const count = this.inferenceResult?.activations?.length ?? 0;
    if (count <= 1) return;
    this.playTimer = window.setInterval(() => {
      this.activeActivationOrder = (this.activeActivationOrder + 1) % count;
    }, 950);
  }

  stopActivationPlayback(): void {
    if (this.playTimer !== null) {
      window.clearInterval(this.playTimer);
      this.playTimer = null;
    }
  }

  logout(): void {
    this.authSvc.logout();
  }

  checkpointConfigText(checkpoint: TrainingCheckpointSummary): string {
    const config = checkpoint.config;
    if (!config) return '超参数 N/A';
    return [
      `优化器 ${config.optimizer ?? 'N/A'}`,
      `LR ${config.learningRate ?? 'N/A'}`,
      `Batch ${config.batchSize ?? 'N/A'}`,
      `Epoch ${config.totalEpochs ?? checkpoint.totalEpochs}`,
      `Loss ${config.lossFunction ?? 'N/A'}`,
      `Scheduler ${config.scheduler ?? 'none'}`
    ].join(' · ');
  }

  checkpointSplitText(checkpoint: TrainingCheckpointSummary): string {
    const split = checkpoint.split;
    if (!split) return '划分 N/A';
    return `${Math.round((split.train ?? 0) * 100)}% / ${Math.round((split.val ?? 0) * 100)}% / ${Math.round((split.test ?? 0) * 100)}%`;
  }

  checkpointLayerPreview(checkpoint: TrainingCheckpointSummary, limit = 8): string[] {
    const summary = checkpoint.layerSummary?.length
      ? checkpoint.layerSummary
      : (checkpoint.layers ?? []).map(layer => `${this.layerTypeLabel(layer.type)} ${layer.name}`);
    return summary.slice(0, limit);
  }

  checkpointLayerMoreCount(checkpoint: TrainingCheckpointSummary, limit = 8): number {
    const count = checkpoint.layerSummary?.length || checkpoint.layers?.length || 0;
    return Math.max(0, count - limit);
  }

  checkpointCreatedText(checkpoint: TrainingCheckpointSummary): string {
    return new Date(checkpoint.createdAt).toLocaleString();
  }

  percent(value: number | null | undefined): string {
    return value === null || value === undefined ? 'N/A' : `${(value * 100).toFixed(1)}%`;
  }

  shapeLabel(shape: number[] | undefined): string {
    return shape?.length ? shape.join(' x ') : 'scalar';
  }

  layerTypeLabel(type: string): string {
    const labels: Record<string, string> = {
      input: '输入层',
      conv2d: '卷积层',
      pool2d: '池化层',
      residual: '残差块',
      flatten: '展平层',
      dense: '全连接层',
      activation: '激活层',
      dropout: 'Dropout',
      output: '输出层'
    };
    return labels[type] ?? type;
  }

  layerIcon(type: string): string {
    const icons: Record<string, string> = {
      input: '⬛',
      conv2d: '⊞',
      pool2d: '⊟',
      residual: '+',
      flatten: '≡',
      dense: '◉',
      activation: 'ƒ',
      dropout: '⊘',
      output: '▶'
    };
    return icons[type] ?? '□';
  }

  layerColor(type: string): string {
    const colors: Record<string, string> = {
      input: '#6366f1',
      conv2d: '#0ea5e9',
      pool2d: '#10b981',
      residual: '#14b8a6',
      flatten: '#f59e0b',
      dense: '#8b5cf6',
      activation: '#ec4899',
      dropout: '#94a3b8',
      output: '#ef4444'
    };
    return colors[type] ?? '#64748b';
  }

  sampleFeatureText(sample: InferenceSampleItem | null): string {
    if (!sample?.featurePreview?.length) return '';
    const values = this.sampleFeatureRows(sample, 4)
      .map(row => `${row.label}: ${row.valueText}`)
      .join('，');
    return `${values}${(sample.featureCount ?? 0) > 4 ? ' ...' : ''}`;
  }

  hasRawData(sample: InferenceSampleItem | null): boolean {
    return !!sample?.rawHeaders?.length || !!sample?.rawValues?.length || !!sample?.rawPreview?.length;
  }

  rawPreviewRows(sample: InferenceSampleItem | null, limit = 6): Array<{ name: string; value: string }> {
    if (!sample) return [];
    if (sample.rawPreview?.length) return sample.rawPreview.slice(0, limit);
    return this.rawDataRows(sample).slice(0, limit);
  }

  rawDataRows(sample: InferenceSampleItem | null): Array<{ name: string; value: string }> {
    const headers = sample?.rawHeaders ?? [];
    const values = sample?.rawValues ?? [];
    const count = Math.max(headers.length, values.length);
    return Array.from({ length: count }, (_, index) => ({
      name: headers[index] || `column ${index + 1}`,
      value: values[index] ?? ''
    }));
  }

  sampleFeatureRows(
    sample: InferenceSampleItem | null,
    limit = 6
  ): Array<{ label: string; value: number; valueText: string; width: number; active: boolean }> {
    const values = sample?.featurePreview ?? [];
    const names = sample?.featureNames ?? [];
    if (!values.length) return [];
    const maxAbs = Math.max(0.000001, ...values.map(value => Math.abs(Number(value) || 0)));
    return values.slice(0, limit).map((raw, index) => {
      const value = Number(raw) || 0;
      return {
        label: names[index] || `特征 ${index + 1}`,
        value,
        valueText: this.formatFeatureValue(value),
        width: Math.max(4, Math.min(100, Math.abs(value) / maxAbs * 100)),
        active: Math.abs(value) > 0.000001
      };
    });
  }

  featureColumnLabels(sample: InferenceSampleItem | null): string[] {
    const names = sample?.featureNames ?? [];
    return Array.from({ length: Math.min(6, sample?.featurePreview?.length ?? 0) }, (_, index) => names[index] || `特征 ${index + 1}`);
  }

  activationBars(activation: SingleInferenceActivation | null): Array<{ index: number; value: number; width: number }> {
    const values = activation?.preview?.values ?? [];
    const max = Math.max(0.000001, ...values.map(value => Math.abs(value)));
    return values.slice(0, 36).map((value, index) => ({
      index,
      value,
      width: Math.max(4, Math.abs(value) / max * 100)
    }));
  }

  private isCompletedCheckpoint(checkpoint: TrainingCheckpointSummary): boolean {
    return checkpoint.status !== 'stopped' && checkpoint.epoch >= checkpoint.totalEpochs;
  }

  private placeCheckpointDetail(target: EventTarget | null): void {
    if (!(target instanceof HTMLElement)) {
      this.checkpointDetailPosition = null;
      return;
    }
    const rect = target.getBoundingClientRect();
    const width = 390;
    const estimatedHeight = 360;
    const gap = 12;
    const padding = 14;
    const rightSideLeft = rect.right + gap;
    const leftSideLeft = rect.left - width - gap;
    const left = rightSideLeft + width <= window.innerWidth - padding
      ? rightSideLeft
      : Math.max(padding, leftSideLeft);
    const maxTop = Math.max(padding, window.innerHeight - estimatedHeight - padding);
    const top = Math.min(Math.max(padding, rect.top - 8), maxTop);
    this.checkpointDetailPosition = { left, top };
  }

  private formatFeatureValue(value: number): string {
    if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.001)) {
      return value.toExponential(2);
    }
    return Number(value.toFixed(4)).toString();
  }

  private buildInferenceLlmContext(): LlmChatContext {
    const checkpoint = this.selectedCheckpoint;
    const sample = this.selectedSample;
    const result = this.inferenceResult;
    const lines = [
      '页面: B 模式 / 单样本推理',
      `当前用户: ${this.authUser?.username ?? '未登录'}`,
      `Checkpoint: ${checkpoint ? `#${checkpoint.id} ${checkpoint.name}` : '未选择'}`
    ];

    if (checkpoint) {
      lines.push(
        `数据集: ${checkpoint.datasetName} (${checkpoint.datasetId})`,
        `训练进度: epoch ${checkpoint.epoch}/${checkpoint.totalEpochs}; status=${checkpoint.status ?? '未知'}`,
        `训练指标: train_loss=${this.llmNumber(checkpoint.trainLoss)}, train_accuracy=${this.percent(checkpoint.trainAccuracy)}, val_loss=${this.llmNumber(checkpoint.valLoss)}, val_accuracy=${this.percent(checkpoint.valAccuracy)}, test_loss=${this.llmNumber(checkpoint.testLoss)}, test_accuracy=${this.percent(checkpoint.testAccuracy)}`,
        `超参数: ${this.checkpointConfigText(checkpoint)}`,
        `数据划分: ${this.checkpointSplitText(checkpoint)}`,
        `网络结构: ${checkpoint.networkDescription || this.checkpointLayerPreview(checkpoint, 30).join(' -> ') || '暂无结构描述'}`
      );
    }

    if (sample) {
      const rawRows = this.rawDataRows(sample).slice(0, 20);
      lines.push(
        '',
        `当前样本: index=${sample.index}; name=${sample.name ?? '未命名'}; true_label=${sample.trueLabel}; shape=${this.shapeLabel(sample.shape)}`,
        `原始数据预览: ${rawRows.length ? rawRows.map(row => `${row.name}=${row.value}`).join('; ') : this.sampleFeatureText(sample) || '无'}`
      );
    } else {
      lines.push('', '当前尚未选择样本。');
    }

    if (result) {
      const prediction = result.prediction;
      lines.push(
        '',
        `预测结果: predicted=${prediction.predictedLabel}; true=${prediction.trueLabel}; confidence=${this.percent(prediction.confidence)}; correct=${prediction.correct ? '是' : '否'}`,
        `Top-K: ${prediction.topK.map(item => `${item.label}=${this.percent(item.probability)}`).join(', ')}`
      );
      for (const activation of result.activations.slice(0, 24)) {
        lines.push(
          `激活层 ${activation.order}: ${activation.layerName} (${this.layerTypeLabel(activation.layerType)}); shape=${this.shapeLabel(activation.shape)}; min=${this.llmNumber(activation.stats.min)}; max=${this.llmNumber(activation.stats.max)}; mean=${this.llmNumber(activation.stats.mean)}; non_zero=${this.percent(activation.stats.nonZeroRatio)}; top_values=${activation.topValues.slice(0, 5).map(item => `${item.index}:${this.llmNumber(item.value)}`).join(', ')}`
        );
      }
      if (result.activations.length > 24) {
        lines.push(`其余 ${result.activations.length - 24} 层激活未展开。`);
      }
    } else {
      lines.push('当前尚未运行推理，没有预测和逐层激活结果。');
    }

    return { text: lines.join('\n'), images: [] };
  }

  private llmNumber(value: number | null | undefined): string {
    return value === null || value === undefined || !Number.isFinite(value)
      ? 'N/A'
      : Number(value.toFixed(6)).toString();
  }
}
