import { CommonModule } from '@angular/common';
import { Component, OnInit, computed, signal } from '@angular/core';
import { ModeCPreviewCanvasComponent } from '../preview-canvas/mode-c-preview-canvas.component';
import {
  ModeCConvChannelExample,
  ModeCConvInputContribution,
  ModeCLayerChannelPreview,
  ModeCNetworkLayer,
  ModeCPoolChannelExample,
  ModeCReluChannelExample
} from '../../models/mode-c.types';
import { ModeCStateService } from '../../services/mode-c-state.service';

type TopologyColumn = ModeCNetworkLayer & {
  x: number;
  previews: ModeCLayerChannelPreview[];
  isActive: boolean;
};

type InputChannelNode = {
  index: number;
  label: string;
  accent: string;
  preview: ModeCLayerChannelPreview | null;
  x: number;
  y: number;
};

type OutputNode = {
  label: string;
  score: number;
  x: number;
  y: number;
  isActive: boolean;
};

type OverlayFocus = {
  sourceLayerId: string | null;
  sourceChannelIndex: number | null;
  targetLayerId: string | null;
  targetChannelIndex: number | null;
};

type ConvOverlayStep = 0 | 1 | 2 | 3;
type ConvOverlayCell = { row: number; col: number } | null;

@Component({
  selector: 'app-mode-c-overview',
  imports: [CommonModule, ModeCPreviewCanvasComponent],
  templateUrl: './mode-c-overview.component.html',
  styleUrl: './mode-c-overview.component.css'
})
export class ModeCOverviewComponent implements OnInit {
  readonly tileSize = 58;
  readonly tileGap = 10;
  readonly convPipelineRowHeight = 116;
  readonly convPipelineBaseOffset = 48;
  readonly overlayDismissed = signal(false);
  readonly convOverlayStep = signal<ConvOverlayStep>(0);
  readonly hoveredConvCell = signal<ConvOverlayCell>(null);
  readonly animatedConvInputIndex = signal(0);
  readonly convScanIndex = signal(0);
  readonly convAutoplay = signal(true);
  readonly reluHoverPoint = signal<ConvOverlayCell>(null);
  readonly poolHoverPoint = signal<ConvOverlayCell>(null);

  readonly inputLayer = computed(() =>
    this.state.networkLayers().find(layer => layer.type === 'input') ?? null
  );

  readonly topologyColumns = computed<TopologyColumn[]>(() => {
    const layers = this.state.networkLayers().filter(layer =>
      layer.type !== 'input' && layer.type !== 'flatten' && layer.type !== 'output'
    );
    const details = this.state.layerDetails();
    const previews = this.state.layerPreviews();

    return layers.map((layer, index) => {
      const detail = details[layer.id] ?? null;
      const channelPreviews = (detail?.channelPreviews ?? []).slice(0, 10);
      const fallbackPreview = previews[layer.id];
      const visiblePreviews = channelPreviews.length
        ? channelPreviews
        : fallbackPreview
          ? [{ index: 0, dataUrl: fallbackPreview.dataUrl, matrix: [[0]], grayscale: false, mean: 0, energy: 0 }]
          : [];

      return {
        ...layer,
        x: 246 + index * 134,
        previews: visiblePreviews,
        isActive: this.state.selectedLayerId() === layer.id
      };
    });
  });

  readonly inputChannels = computed<InputChannelNode[]>(() => {
    const previews = (this.inputLayer() ? this.state.layerDetails()[this.inputLayer()!.id]?.channelPreviews : []) ?? [];
    const visible = previews.slice(0, 3);
    const labels = [
      { label: '红色通道', accent: '#dc2626' },
      { label: '绿色通道', accent: '#16a34a' },
      { label: '蓝色通道', accent: '#2563eb' }
    ];

    return labels.map((meta, index) => ({
      index,
      label: meta.label,
      accent: meta.accent,
      preview: visible[index] ?? null,
      x: 62,
      y: 182 + index * 166
    }));
  });

  readonly outputNodes = computed<OutputNode[]>(() => {
    const prediction = this.state.currentSamplePrediction();
    const activeLabel = prediction?.label ?? '';
    return (prediction?.topClasses ?? []).slice(0, 10).map((candidate, index) => ({
      label: candidate.label,
      score: candidate.score,
      x: this.outputColumnX(),
      y: 126 + index * 62,
      isActive: candidate.label === activeLabel
    }));
  });

  readonly outputColumnX = computed(() => 246 + this.topologyColumns().length * 134 + 124);
  readonly boardHeight = computed(() => {
    const maxFeatureTiles = Math.max(
      this.inputChannels().length,
      ...this.topologyColumns().map(layer => layer.previews.length)
    );
    const featureBottom = maxFeatureTiles > 0
      ? this.getTileY(maxFeatureTiles - 1) + this.tileSize + 44
      : 0;
    const outputBottom = this.outputNodes().length
      ? 126 + (this.outputNodes().length - 1) * 62 + 52
      : 0;

    return Math.max(920, featureBottom + 32, outputBottom + 24);
  });

  readonly boardWidth = computed(() => this.outputColumnX() + 190);
  readonly activeConvExample = computed<ModeCConvChannelExample | null>(() => {
    const layer = this.state.selectedLayer();
    if (!layer || layer.type !== 'conv') {
      return null;
    }
    const examples = this.state.selectedLayerDetail()?.convExamples ?? [];
    return examples.find(example => example.outputChannelIndex === this.state.selectedChannelIndex()) ?? examples[0] ?? null;
  });
  readonly activeConvSourcePreview = computed<ModeCLayerChannelPreview | null>(() => {
    const example = this.activeConvExample();
    const previousLayer = this.state.previousLayer();
    if (!example || !previousLayer) {
      return null;
    }
    const previews = this.state.layerDetails()[previousLayer.id]?.channelPreviews ?? [];
    return previews.find(preview => preview.index === example.inputChannelIndex) ?? null;
  });
  readonly activeConvTargetPreview = computed<ModeCLayerChannelPreview | null>(() => {
    const example = this.activeConvExample();
    const layer = this.state.selectedLayer();
    if (!example || !layer) {
      return null;
    }
    const previews = this.state.layerDetails()[layer.id]?.channelPreviews ?? [];
    return previews.find(preview => preview.index === example.outputChannelIndex) ?? null;
  });
  readonly animatedConvContribution = computed<ModeCConvInputContribution | null>(() => {
    const example = this.activeConvExample();
    if (!example) {
      return null;
    }
    return example.inputContributions[this.animatedConvInputIndex()] ?? example.inputContributions[0] ?? null;
  });
  readonly animatedConvSourcePreview = computed<ModeCLayerChannelPreview | null>(() => {
    const contribution = this.animatedConvContribution();
    const previousLayer = this.state.previousLayer();
    if (!contribution || !previousLayer) {
      return null;
    }
    const previews = this.state.layerDetails()[previousLayer.id]?.channelPreviews ?? [];
    return previews.find(preview => preview.index === contribution.inputChannelIndex) ?? null;
  });
  readonly convScanPositions = computed(() => {
    const example = this.activeConvExample();
    const targetMatrix = this.activeConvTargetPreview()?.matrix;
    if (!example || !targetMatrix?.length || !targetMatrix[0]?.length) {
      return [{ row: example?.row ?? 0, col: example?.col ?? 0 }];
    }

    const radius = 1;
    const positions: Array<{ row: number; col: number }> = [];
    for (let row = Math.max(0, example.row - radius); row <= Math.min(targetMatrix.length - 1, example.row + radius); row += 1) {
      for (let col = Math.max(0, example.col - radius); col <= Math.min(targetMatrix[0].length - 1, example.col + radius); col += 1) {
        positions.push({ row, col });
      }
    }

    return positions.length ? positions : [{ row: example.row, col: example.col }];
  });
  readonly activeConvScanPoint = computed(() => {
    const positions = this.convScanPositions();
    return positions[this.convScanIndex()] ?? positions[0] ?? { row: 0, col: 0 };
  });
  readonly activeConvDynamicPatch = computed<number[][]>(() => {
    const contribution = this.animatedConvContribution();
    const sourceMatrix = this.animatedConvSourcePreview()?.matrix;
    const point = this.activeConvScanPoint();
    if (!contribution || !sourceMatrix?.length || !sourceMatrix[0]?.length) {
      return contribution?.patch ?? [];
    }
    return this.extractMatrixPatch(sourceMatrix, point.row, point.col, contribution.patch.length || contribution.kernel.length || 3);
  });
  readonly activeConvDynamicProducts = computed<number[][]>(() => {
    const contribution = this.animatedConvContribution();
    const patch = this.activeConvDynamicPatch();
    if (!contribution) {
      return [];
    }
    return patch.map((patchRow, rowIndex) =>
      patchRow.map((value, colIndex) => value * (contribution.kernel[rowIndex]?.[colIndex] ?? 0))
    );
  });
  readonly activeConvDynamicWeightedSum = computed(() =>
    this.activeConvDynamicProducts().flat().reduce((sum, value) => sum + value, 0)
  );
  readonly activeConvDynamicOutputValue = computed(() => {
    const point = this.activeConvScanPoint();
    const matrix = this.activeConvTargetPreview()?.matrix;
    return matrix?.[point.row]?.[point.col] ?? this.activeConvExample()?.outputValue ?? 0;
  });
  readonly activeConvDynamicContributionValues = computed(() => {
    const example = this.activeConvExample();
    const sourceLayer = this.state.previousLayer();
    const point = this.activeConvScanPoint();
    if (!example || !sourceLayer) {
      return new Map<number, number>();
    }

    const previews = this.state.layerDetails()[sourceLayer.id]?.channelPreviews ?? [];
    const values = new Map<number, number>();
    for (const contribution of example.inputContributions) {
      const sourcePreview = previews.find(preview => preview.index === contribution.inputChannelIndex);
      if (!sourcePreview?.matrix?.length) {
        values.set(contribution.inputChannelIndex, contribution.weightedSum);
        continue;
      }

      const patch = this.extractMatrixPatch(
        sourcePreview.matrix,
        point.row,
        point.col,
        contribution.patch.length || contribution.kernel.length || 3
      );
      const products = patch.map((patchRow, rowIndex) =>
        patchRow.map((value, colIndex) => value * (contribution.kernel[rowIndex]?.[colIndex] ?? 0))
      );
      const weightedSum = products.flat().reduce((sum, value) => sum + value, 0);
      values.set(contribution.inputChannelIndex, weightedSum);
    }

    return values;
  });
  readonly convContributionTotal = computed(() => {
    const example = this.activeConvExample();
    return example?.inputContributions.reduce((sum, item) => sum + item.weightedSum, 0) ?? 0;
  });
  readonly convAccumulatedTotal = computed(() => {
    const example = this.activeConvExample();
    if (!example?.inputContributions.length) {
      return 0;
    }

    const maxIndex = Math.min(this.animatedConvInputIndex(), example.inputContributions.length - 1);
    return example.inputContributions
      .slice(0, maxIndex + 1)
      .reduce((sum, item) => sum + item.weightedSum, 0);
  });
  readonly convAccumulatedOutput = computed(() => {
    const example = this.activeConvExample();
    if (!example) {
      return 0;
    }
    return this.convAccumulatedTotal() + example.bias;
  });
  readonly activeReluExample = computed<ModeCReluChannelExample | null>(() => {
    const layer = this.state.selectedLayer();
    if (!layer || layer.type !== 'relu') {
      return null;
    }
    const examples = this.state.selectedLayerDetail()?.reluExamples ?? [];
    return examples.find(example => example.channelIndex === this.state.selectedChannelIndex()) ?? examples[0] ?? null;
  });
  readonly activePoolExample = computed<ModeCPoolChannelExample | null>(() => {
    const layer = this.state.selectedLayer();
    if (!layer || layer.type !== 'pool') {
      return null;
    }
    const examples = this.state.selectedLayerDetail()?.poolExamples ?? [];
    return examples.find(example => example.channelIndex === this.state.selectedChannelIndex()) ?? examples[0] ?? null;
  });
  readonly showConvOverlay = computed(() => !this.overlayDismissed() && !!this.activeConvExample());
  readonly showReluOverlay = computed(() => !this.overlayDismissed() && !!this.activeReluExample());
  readonly showPoolOverlay = computed(() => !this.overlayDismissed() && !!this.activePoolExample());
  readonly activeReluBeforePreview = computed<ModeCLayerChannelPreview | null>(() => {
    const example = this.activeReluExample();
    const previousLayer = this.state.previousLayer();
    if (!example || !previousLayer) {
      return null;
    }
    const previews = this.state.layerDetails()[previousLayer.id]?.channelPreviews ?? [];
    return previews.find(preview => preview.index === example.channelIndex) ?? null;
  });
  readonly activeReluAfterPreview = computed<ModeCLayerChannelPreview | null>(() => {
    const example = this.activeReluExample();
    const layer = this.state.selectedLayer();
    if (!example || !layer) {
      return null;
    }
    const previews = this.state.layerDetails()[layer.id]?.channelPreviews ?? [];
    return previews.find(preview => preview.index === example.channelIndex) ?? null;
  });
  readonly activeReluHoverValues = computed(() => {
    const point = this.reluHoverPoint();
    const before = this.activeReluBeforePreview()?.matrix;
    const after = this.activeReluAfterPreview()?.matrix;
    if (!point || !before?.length || !after?.length) {
      return null;
    }

    const row = Math.min(point.row, before.length - 1, after.length - 1);
    const col = Math.min(point.col, before[0].length - 1, after[0].length - 1);
    return {
      row,
      col,
      before: before[row]?.[col] ?? 0,
      after: after[row]?.[col] ?? 0
    };
  });
  readonly activePoolInputPreview = computed<ModeCLayerChannelPreview | null>(() => {
    const example = this.activePoolExample();
    const previousLayer = this.state.previousLayer();
    if (!example || !previousLayer) {
      return null;
    }
    const previews = this.state.layerDetails()[previousLayer.id]?.channelPreviews ?? [];
    return previews.find(preview => preview.index === example.channelIndex) ?? null;
  });
  readonly activePoolOutputPreview = computed<ModeCLayerChannelPreview | null>(() => {
    const example = this.activePoolExample();
    const layer = this.state.selectedLayer();
    if (!example || !layer) {
      return null;
    }
    const previews = this.state.layerDetails()[layer.id]?.channelPreviews ?? [];
    return previews.find(preview => preview.index === example.channelIndex) ?? null;
  });
  readonly activePoolHoverValues = computed(() => {
    const point = this.poolHoverPoint();
    const input = this.activePoolInputPreview()?.matrix;
    const output = this.activePoolOutputPreview()?.matrix;
    if (!point || !input?.length || !output?.length) {
      return null;
    }

    const inputRow = Math.min(point.row, input.length - 1);
    const inputCol = Math.min(point.col, input[0].length - 1);
    const outputRow = Math.min(Math.floor(inputRow / 2), output.length - 1);
    const outputCol = Math.min(Math.floor(inputCol / 2), output[0].length - 1);

    return {
      inputRow,
      inputCol,
      outputRow,
      outputCol,
      inputValue: input[inputRow]?.[inputCol] ?? 0,
      outputValue: output[outputRow]?.[outputCol] ?? 0
    };
  });
  readonly activePoolWindowValues = computed(() => {
    const example = this.activePoolExample();
    const input = this.activePoolInputPreview()?.matrix;
    const output = this.activePoolOutputPreview()?.matrix;
    if (!example || !input?.length || !output?.length) {
      return null;
    }

    const hover = this.activePoolHoverValues();
    const outputRow = hover?.outputRow ?? example.row;
    const outputCol = hover?.outputCol ?? example.col;
    const startRow = Math.min(outputRow * 2, Math.max(0, input.length - 2));
    const startCol = Math.min(outputCol * 2, Math.max(0, input[0].length - 2));
    const values = [
      input[startRow]?.[startCol] ?? 0,
      input[startRow]?.[startCol + 1] ?? 0,
      input[startRow + 1]?.[startCol] ?? 0,
      input[startRow + 1]?.[startCol + 1] ?? 0
    ];
    const maxValue = Math.max(...values);
    const maxIndex = values.findIndex(value => value === maxValue);

    return {
      startRow,
      startCol,
      outputRow,
      outputCol,
      values,
      maxValue,
      outputValue: output[outputRow]?.[outputCol] ?? example.maxValue,
      maxIndex
    };
  });
  readonly hasActiveOverlay = computed(() =>
    this.showConvOverlay() || this.showReluOverlay() || this.showPoolOverlay()
  );
  readonly convStepItems = [
    { id: 0 as ConvOverlayStep, label: '输入来源', note: '找到产生当前响应的输入通道。' },
    { id: 1 as ConvOverlayStep, label: '局部块 × 卷积核', note: '将当前局部输入块与卷积核权重一一对齐。' },
    { id: 2 as ConvOverlayStep, label: '逐项乘积', note: '观察每个位置上的逐元素乘法结果。' },
    { id: 3 as ConvOverlayStep, label: '累加', note: '将乘积求和，加上偏置，并写入输出值。' }
  ] as const;
  readonly activeConvStepMeta = computed(() => this.convStepItems[this.convOverlayStep()]);
  readonly activeConvCellDetail = computed(() => {
    const contribution = this.animatedConvContribution();
    const cell = this.hoveredConvCell();
    const patch = this.activeConvDynamicPatch();
    const products = this.activeConvDynamicProducts();
    if (!contribution || !cell) {
      return null;
    }

    return {
      row: cell.row,
      col: cell.col,
      patchValue: patch[cell.row]?.[cell.col] ?? 0,
      kernelValue: contribution.kernel[cell.row]?.[cell.col] ?? 0,
      productValue: products[cell.row]?.[cell.col] ?? 0
    };
  });
  readonly overlayFocus = computed<OverlayFocus>(() => {
    if (this.showConvOverlay()) {
      const example = this.activeConvExample();
      const animatedContribution = this.animatedConvContribution();
      const previousLayer = this.state.previousLayer();
      const currentLayer = this.state.selectedLayer();
      const step = this.convOverlayStep();
      return {
        sourceLayerId: step <= 2 ? previousLayer?.id ?? null : null,
        sourceChannelIndex: step <= 2 ? (animatedContribution?.inputChannelIndex ?? example?.inputChannelIndex ?? null) : null,
        targetLayerId: step >= 1 ? currentLayer?.id ?? null : null,
        targetChannelIndex: step >= 1 ? example?.outputChannelIndex ?? null : null
      };
    }

    if (this.showReluOverlay() || this.showPoolOverlay()) {
      const previousLayer = this.state.previousLayer();
      const currentLayer = this.state.selectedLayer();
      const channelIndex = this.state.selectedChannelIndex();
      return {
        sourceLayerId: previousLayer?.id ?? null,
        sourceChannelIndex: channelIndex,
        targetLayerId: currentLayer?.id ?? null,
        targetChannelIndex: channelIndex
      };
    }

    return {
      sourceLayerId: null,
      sourceChannelIndex: null,
      targetLayerId: null,
      targetChannelIndex: null
    };
  });
  readonly overlayTitle = computed(() => {
    if (this.showConvOverlay()) return '卷积过程拆解';
    if (this.showReluOverlay()) return 'ReLU 激活';
    if (this.showPoolOverlay()) return '最大池化步骤';
    return '';
  });
  readonly overlayHint = computed(() => {
    if (this.showConvOverlay()) {
      return '从一个输出通道反向追踪到输入局部块、卷积核和加权求和过程。';
    }
    if (this.showReluOverlay()) {
      return '对比同一通道在非线性激活前后的变化，观察负响应如何被抑制。';
    }
    if (this.showPoolOverlay()) {
      return '观察在空间下采样后保留下来的 2×2 窗口最大响应。';
    }
    return '';
  });

  readonly topologyLinks = computed(() => {
    const paths: string[] = [];
    const columns = this.topologyColumns();
    const inputs = this.inputChannels();
    const outputs = this.outputNodes();

    if (columns.length) {
      const first = columns[0];
      for (const input of inputs) {
        for (let index = 0; index < Math.min(first.previews.length, 10); index += 1) {
          paths.push(this.buildCurve(input.x + this.tileSize, input.y + this.tileSize / 2, first.x, this.getTileY(index) + this.tileSize / 2));
        }
      }
    }

    for (let columnIndex = 0; columnIndex < columns.length - 1; columnIndex += 1) {
      const current = columns[columnIndex];
      const next = columns[columnIndex + 1];
      const currentCount = Math.min(current.previews.length, 10);
      const nextCount = Math.min(next.previews.length, 10);

      for (let left = 0; left < currentCount; left += 1) {
        for (let right = 0; right < nextCount; right += 1) {
          paths.push(this.buildCurve(current.x + this.tileSize, this.getTileY(left) + this.tileSize / 2, next.x, this.getTileY(right) + this.tileSize / 2));
        }
      }
    }

    if (columns.length && outputs.length) {
      const last = columns[columns.length - 1];
      const lastCount = Math.min(last.previews.length, 10);
      for (let left = 0; left < lastCount; left += 1) {
        for (const output of outputs) {
          paths.push(this.buildCurve(last.x + this.tileSize, this.getTileY(left) + this.tileSize / 2, output.x, output.y + 8));
        }
      }
    }

    return paths;
  });

  constructor(readonly state: ModeCStateService) {}

  ngOnInit(): void {
    void this.state.initializeNetworkLayers();
    if (typeof window !== 'undefined') {
      window.setInterval(() => {
        const example = this.activeConvExample();
        if (!example || !example.inputContributions.length || !this.showConvOverlay() || !this.convAutoplay()) {
          return;
        }

        if (this.convOverlayStep() <= 2 && this.convScanPositions().length > 1) {
          const nextScanIndex = this.convScanIndex() + 1;
          if (nextScanIndex >= this.convScanPositions().length) {
            this.convScanIndex.set(0);
            this.animatedConvInputIndex.update(index => (index + 1) % example.inputContributions.length);
          } else {
            this.convScanIndex.set(nextScanIndex);
          }
          return;
        }

        this.animatedConvInputIndex.update(index => (index + 1) % example.inputContributions.length);
      }, 520);
    }
  }

  selectSample(sampleId: string): void {
    this.state.setCurrentSample(sampleId);
    this.state.setActiveFocus('overview');
  }

  selectLayer(layerId: string): void {
    this.state.setSelectedLayer(layerId);
    this.convOverlayStep.set(0);
    this.hoveredConvCell.set(null);
    this.animatedConvInputIndex.set(0);
    this.convScanIndex.set(0);
    this.convAutoplay.set(true);
    this.overlayDismissed.set(false);
  }

  selectLayerChannel(layerId: string, channelIndex: number): void {
    this.state.setSelectedLayer(layerId);
    this.state.setSelectedChannel(channelIndex);
    this.convOverlayStep.set(0);
    this.hoveredConvCell.set(null);
    this.animatedConvInputIndex.set(0);
    this.convScanIndex.set(0);
    this.convAutoplay.set(true);
    this.overlayDismissed.set(false);
  }

  focusTopic(topicId: string): void {
    this.state.setSelectedTopic(topicId);
  }

  trackLink(index: number): number {
    return index;
  }

  getTileY(index: number): number {
    return 118 + index * (this.tileSize + this.tileGap);
  }

  getOutputBarWidth(score: number): number {
    return Math.max(6, Math.min(90, score * 100));
  }

  getConvWindowInsetPercent(axis: 'row' | 'col'): number {
    const point = this.activeConvScanPoint();
    const previous = this.state.previousLayer();
    const contribution = this.animatedConvContribution();
    if (!point || !previous || !contribution) {
      return 8;
    }

    const kernelSize = contribution.patch.length || contribution.kernel.length || 3;
    const sourceSize = Math.max(previous.spatialSize, kernelSize + 1);
    const maxOffset = Math.max(1, sourceSize - kernelSize);
    const currentOffset = axis === 'row' ? point.row : point.col;
    const normalized = Math.max(0, Math.min(1, currentOffset / maxOffset));
    return 8 + normalized * 56;
  }

  getConvWindowSizePercent(): number {
    const contribution = this.animatedConvContribution();
    const previous = this.state.previousLayer();
    if (!contribution || !previous) {
      return 18;
    }

    const kernelSize = contribution.patch.length || contribution.kernel.length || 3;
    const sourceSize = Math.max(previous.spatialSize, kernelSize);
    return (kernelSize / sourceSize) * 100;
  }

  getConvOutputGridStepPercent(): number {
    const matrix = this.activeConvTargetPreview()?.matrix;
    if (!matrix?.length) {
      return 10;
    }
    return 100 / matrix.length;
  }

  getConvOutputCellInsetPercent(axis: 'row' | 'col'): number {
    const point = this.activeConvScanPoint();
    const matrix = this.activeConvTargetPreview()?.matrix;
    if (!matrix?.length || !matrix[0]?.length) {
      return 0;
    }
    const size = axis === 'row' ? matrix.length : matrix[0].length;
    const current = axis === 'row' ? point.row : point.col;
    return (current / size) * 100;
  }

  getConvOutputCellSizePercent(axis: 'row' | 'col'): number {
    const matrix = this.activeConvTargetPreview()?.matrix;
    if (!matrix?.length || !matrix[0]?.length) {
      return 8;
    }
    return axis === 'row' ? 100 / matrix.length : 100 / matrix[0].length;
  }

  getConvGridStepPercent(): number {
    const previous = this.state.previousLayer();
    if (!previous?.spatialSize) {
      return 10;
    }
    return 100 / previous.spatialSize;
  }

  getConvPipelineOffsetPx(): number {
    const count = this.activeConvExample()?.inputContributions.length ?? 0;
    if (!count) {
      return this.convPipelineBaseOffset;
    }
    return this.convPipelineBaseOffset + ((count - 1) * this.convPipelineRowHeight) / 2;
  }

  getContributionFillPercent(value: number): number {
    const total = Math.max(
      ...((this.activeConvExample()?.inputContributions ?? []).map(item => Math.abs(item.weightedSum))),
      0.0001
    );
    return Math.max(8, Math.min(100, (Math.abs(value) / total) * 100));
  }

  getConvContributionValue(channelIndex: number): number {
    return this.activeConvDynamicContributionValues().get(channelIndex) ?? 0;
  }

  getConvContributionSourcePreview(channelIndex: number): ModeCLayerChannelPreview | null {
    const previousLayer = this.state.previousLayer();
    if (!previousLayer) {
      return null;
    }
    const previews = this.state.layerDetails()[previousLayer.id]?.channelPreviews ?? [];
    return previews.find(preview => preview.index === channelIndex) ?? null;
  }

  isOverlaySource(layerId: string, channelIndex: number): boolean {
    const focus = this.overlayFocus();
    return focus.sourceLayerId === layerId && focus.sourceChannelIndex === channelIndex;
  }

  isOverlayTarget(layerId: string, channelIndex: number): boolean {
    const focus = this.overlayFocus();
    return focus.targetLayerId === layerId && focus.targetChannelIndex === channelIndex;
  }

  formatPercent(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  formatSigned(value: number, digits = 3): string {
    const formatted = value.toFixed(digits);
    return value > 0 ? `+${formatted}` : formatted;
  }

  getMatrixMin(matrix: number[][]): number {
    const values = matrix.flat().filter(value => Number.isFinite(value));
    return values.length ? Math.min(...values) : 0;
  }

  getMatrixMax(matrix: number[][]): number {
    const values = matrix.flat().filter(value => Number.isFinite(value));
    return values.length ? Math.max(...values) : 0;
  }

  getMatrixSample(matrix: number[][]): number {
    if (!matrix.length || !matrix[0]?.length) return 0;
    return matrix[Math.floor(matrix.length / 2)]?.[Math.floor(matrix[0].length / 2)] ?? 0;
  }

  closeOverlay(): void {
    this.overlayDismissed.set(true);
  }

  setConvOverlayStep(step: ConvOverlayStep): void {
    this.convOverlayStep.set(step);
    this.hoveredConvCell.set(null);
    this.convScanIndex.set(0);
  }

  setAnimatedConvInput(index: number): void {
    this.animatedConvInputIndex.set(index);
    this.convScanIndex.set(0);
    this.convAutoplay.set(false);
    this.hoveredConvCell.set(null);
  }

  toggleConvAutoplay(): void {
    this.convAutoplay.update(value => !value);
  }

  previousConvInputContribution(): void {
    const example = this.activeConvExample();
    if (!example?.inputContributions.length) {
      return;
    }
    this.convAutoplay.set(false);
    this.animatedConvInputIndex.update(index =>
      (index - 1 + example.inputContributions.length) % example.inputContributions.length
    );
    this.convScanIndex.set(0);
    this.hoveredConvCell.set(null);
  }

  nextConvInputContribution(): void {
    const example = this.activeConvExample();
    if (!example?.inputContributions.length) {
      return;
    }
    this.convAutoplay.set(false);
    this.animatedConvInputIndex.update(index => (index + 1) % example.inputContributions.length);
    this.convScanIndex.set(0);
    this.hoveredConvCell.set(null);
  }

  previousConvOverlayStep(): void {
    this.convOverlayStep.update(step => Math.max(0, step - 1) as ConvOverlayStep);
  }

  nextConvOverlayStep(): void {
    this.convOverlayStep.update(step => Math.min(3, step + 1) as ConvOverlayStep);
  }

  isConvOverlayStep(step: ConvOverlayStep): boolean {
    return this.convOverlayStep() === step;
  }

  setHoveredConvCell(row: number, col: number): void {
    this.hoveredConvCell.set({ row, col });
  }

  clearHoveredConvCell(): void {
    this.hoveredConvCell.set(null);
  }

  isHoveredConvCell(row: number, col: number): boolean {
    const cell = this.hoveredConvCell();
    return !!cell && cell.row === row && cell.col === col;
  }

  isAnimatedConvContribution(channelIndex: number): boolean {
    return this.animatedConvContribution()?.inputChannelIndex === channelIndex;
  }

  isAccumulatedConvContribution(channelIndex: number): boolean {
    const example = this.activeConvExample();
    if (!example) {
      return false;
    }

    const index = example.inputContributions.findIndex(item => item.inputChannelIndex === channelIndex);
    return index >= 0 && index <= this.animatedConvInputIndex();
  }

  setReluHoverFromEvent(event: MouseEvent): void {
    const point = this.getMatrixPointFromEvent(event, this.activeReluBeforePreview()?.matrix ?? null);
    this.reluHoverPoint.set(point);
  }

  clearReluHover(): void {
    this.reluHoverPoint.set(null);
  }

  setPoolHoverFromEvent(event: MouseEvent): void {
    const point = this.getMatrixPointFromEvent(event, this.activePoolInputPreview()?.matrix ?? null);
    this.poolHoverPoint.set(point);
  }

  clearPoolHover(): void {
    this.poolHoverPoint.set(null);
  }

  getHoverMarkerTopPercent(point: ConvOverlayCell | null, matrix: number[][] | null | undefined): number {
    if (!point || !matrix?.length) {
      return 0;
    }
    return ((point.row + 0.5) / matrix.length) * 100;
  }

  getHoverMarkerLeftPercent(point: ConvOverlayCell | null, matrix: number[][] | null | undefined): number {
    if (!point || !matrix?.[0]?.length) {
      return 0;
    }
    return ((point.col + 0.5) / matrix[0].length) * 100;
  }

  getPoolOutputMarkerTopPercent(): number {
    const values = this.activePoolHoverValues();
    const matrix = this.activePoolOutputPreview()?.matrix;
    if (!values || !matrix?.length) {
      return 0;
    }
    return ((values.outputRow + 0.5) / matrix.length) * 100;
  }

  getPoolOutputMarkerLeftPercent(): number {
    const values = this.activePoolHoverValues();
    const matrix = this.activePoolOutputPreview()?.matrix;
    if (!values || !matrix?.[0]?.length) {
      return 0;
    }
    return ((values.outputCol + 0.5) / matrix[0].length) * 100;
  }

  private buildCurve(startX: number, startY: number, endX: number, endY: number): string {
    const curveX = startX + (endX - startX) * 0.52;
    return `M ${startX} ${startY} C ${curveX} ${startY}, ${curveX} ${endY}, ${endX} ${endY}`;
  }

  private extractMatrixPatch(matrix: number[][], startRow: number, startCol: number, kernelSize: number): number[][] {
    const patch: number[][] = [];
    for (let row = 0; row < kernelSize; row += 1) {
      patch.push([]);
      for (let col = 0; col < kernelSize; col += 1) {
        patch[row].push(matrix[startRow + row]?.[startCol + col] ?? 0);
      }
    }
    return patch;
  }

  private getMatrixPointFromEvent(event: MouseEvent, matrix: number[][] | null): ConvOverlayCell {
    if (!matrix?.length || !matrix[0]?.length) {
      return null;
    }

    const element = event.currentTarget as HTMLElement | null;
    if (!element) {
      return null;
    }

    const rect = element.getBoundingClientRect();
    if (!rect.width || !rect.height) {
      return null;
    }

    const x = Math.max(0, Math.min(rect.width - 1, event.clientX - rect.left));
    const y = Math.max(0, Math.min(rect.height - 1, event.clientY - rect.top));
    const col = Math.min(matrix[0].length - 1, Math.floor((x / rect.width) * matrix[0].length));
    const row = Math.min(matrix.length - 1, Math.floor((y / rect.height) * matrix.length));
    return { row, col };
  }
}
