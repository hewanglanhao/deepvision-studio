import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { Subscription } from 'rxjs';
import { NetworkOverviewComponent } from '../../components/network-overview.component';
import { PlatformTopbarComponent } from '../../components/platform-topbar.component';
import { AuthUser } from '../../models/auth.models';
import { ForwardRecordDetail, ForwardRecordSummary, ForwardRecordSnapshot } from '../../models/forward-record.models';
import { AuthService } from '../../services/auth.service';
import { ForwardRecordService } from '../../services/forward-record.service';
import { ForwardBackendService } from '../../services/forward-backend.service';
import { TrainingCheckpointSummary, TrainingLog, TrainingRuntimeService, TrainingTestResult } from '../../services/training-runtime.service';
import { TrainingDatasetApiService } from '../../services/training-dataset-api.service';
import { SimEngine } from '../../sim-engine';
import {
  AppMode, Connection, DataSample, DatasetImportDraft, ExperimentResult,
  ConvKernelSpec,
  ForwardInputAsset, ForwardLayerResult, ForwardPassResult,
  ForwardTensor, ImagePreviewItem, InputLayer, LabelDistributionItem, LayerType,
  LayerValidationIssue, MetricPoint, ModelTemplate, NetworkLayer, PointPreviewItem,
  PresetTask, TablePreview, TensorShape, TrainingConfig, TrainingDatasetDetail, TrainingDatasetKind,
  TrainingDatasetOption
} from '../../sim-models';

/** 上传图片显示预览最大边长（保留较高分辨率） */
const MAX_IMAGE_SIDE = 640;
/** 图片解码超时 ms */
const IMAGE_DECODE_TIMEOUT = 5000;
/** DOM 像素网格只用于教学预览，避免大图生成海量节点。计算张量不受这个限制。 */
const MAX_PREVIEW_GRID_SIDE = 56;

export interface KernelPreset {
  label: string;
  matrix: number[][];
}

interface ChannelPreviewItem {
  channel: number;
  width: number;
  height: number;
  values: number[];
}

interface LocalImageSample {
  id: string;
  name: string;
  label: string;
  category: string;
  url: string;
  source?: string;
}

export const KERNEL_PRESETS: KernelPreset[] = [
  { label: 'Identity',     matrix: [[0,0,0],[0,1,0],[0,0,0]] },
  { label: 'Edge Detect',  matrix: [[-1,-1,-1],[-1,8,-1],[-1,-1,-1]] },
  { label: 'Sharpen',      matrix: [[0,-1,0],[-1,5,-1],[0,-1,0]] },
  { label: 'Box Blur',     matrix: [[1/9,1/9,1/9],[1/9,1/9,1/9],[1/9,1/9,1/9]] },
  { label: 'Gaussian',     matrix: [[1/16,2/16,1/16],[2/16,4/16,2/16],[1/16,2/16,1/16]] },
  { label: 'Emboss',       matrix: [[-2,-1,0],[-1,1,1],[0,1,2]] },
  { label: 'Sobel X',      matrix: [[-1,0,1],[-2,0,2],[-1,0,1]] },
  { label: 'Sobel Y',      matrix: [[-1,-2,-1],[0,0,0],[1,2,1]] },
];

const DATASET_COLORS = ['#2563eb', '#059669', '#d97706', '#dc2626', '#7c3aed', '#0891b2', '#be123c', '#4b5563'];
type TrainingChartMetric = 'loss' | 'valLoss' | 'accuracy' | 'valAccuracy' | 'lr' | 'gradientNorm';
type TrainingChartFormat = 'number' | 'percent' | 'lr';

@Component({
  selector: 'app-mode-b-page',
  imports: [CommonModule, FormsModule, DecimalPipe, RouterModule, NetworkOverviewComponent, PlatformTopbarComponent],
  templateUrl: './mode-b-page.component.html',
  styleUrl: './mode-b-page.component.css'
})
export class ModeBPageComponent implements OnInit, OnDestroy {
  readonly topbarModeLabel = '\u6a21\u5f0f B';
  readonly topbarModeTitle = '\u6a21\u578b\u8bad\u7ec3\u5de5\u4f5c\u53f0';
  mode: AppMode = 'training';
  showSamplePicker = false;
  authUser: AuthUser | null = null;
  showAuthModal = false;
  authMode: 'login' | 'register' = 'login';
  authDraft = { username: '', password: '', displayName: '' };
  authBusy = false;
  authError = '';

  get topbarStatusPills(): string[] {
    return [`${this.layerCount} 层`, `${this.parameterCount.toLocaleString()} 参数`];
  }

  showSaveRecordModal = false;
  showRecordDrawer = false;
  recordNameDraft = '';
  recordBusy = false;
  recordError = '';
  recordSuccess = '';
  forwardRecords: ForwardRecordSummary[] = [];
  imageViewer: { open: boolean; title: string; url: string; meta: string } = {
    open: false,
    title: '',
    url: '',
    meta: ''
  };

  modelTemplates: ModelTemplate[] = SimEngine.templates();
  selectedTemplateId = 'cnn-classic';
  layers: NetworkLayer[] = [];
  connections: Connection[] = [];
  nextLayerId = 1;
  selectedLayerId = -1;

  datasets: Record<string, DataSample[]> = {};
  selectedDataset = 'Animal';
  selectedSampleId = 1;
  uploadComputeProfile: 'fast' | 'balanced' | 'quality' | 'original' = 'balanced';
  uploadedImageUrl = '';
  uploadError = '';
  localImageSamples: LocalImageSample[] = [];
  selectedLocalImageId = '';
  localImageError = '';
  private uploadedImageData: ImageData | null = null;
  private localImageData: ImageData | null = null;
  private localImagePreviewUrl = '';
  currentInputAsset: ForwardInputAsset | null = null;

  forwardResult: ForwardPassResult | null = null;
  forwardLayerShapeMap: Record<number, string> = {};
  forwardBusy = false;
  forwardBackendError = '';
  autoForwardCompute = false;
  pendingForwardChanges = false;
  forwardStatusMessage = '';

  trainingConfig: TrainingConfig & { lossFunction: string } = {
    batchSize: 32, totalEpochs: 20, learningRate: 0.001,
    optimizer: 'Adam', scheduler: 'none', lrDecay: 0.9,
    lossFunction: 'cross_entropy'
  };
  trainingStatus = 'idle';
  trainingEpoch = 0;
  trainingLr = 0.001;
  trainingLoss = 1.7;
  trainingValLoss = 1.78;
  trainingAcc = 0.2;
  trainingValAcc = 0.18;
  trainingGradientNorm = 1.2;
  trainingWeightMean = 0;
  trainingWeightStd = 0.16;
  trainingElapsedSeconds = 0;
  trainingEtaSeconds = 0;
  trainingCurrentBatchValue = 0;
  trainingTotalBatchesValue = 0;
  trainingHistory: MetricPoint[] = [];
  trainingLogs: TrainingLog[] = [];
  trainingTestResult: TrainingTestResult | null = null;
  trainingCheckpoints: TrainingCheckpointSummary[] = [];
  selectedCheckpointId: number | null = null;
  checkpointBusy = false;
  checkpointError = '';

  selectedTaskId = 'mnist-classify';
  experimentResults: ExperimentResult[] = [];
  readonly kernelPresets = KERNEL_PRESETS;
  selectedKernelOutChannel = 0;
  selectedKernelInChannel = 0;
  showChannelModal = false;
  channelModalTitle = '';
  channelModalPreviews: ChannelPreviewItem[] = [];

  readonly presetTasks: PresetTask[] = [
    { id: 'mnist-classify',  name: '手写数字识别',   type: 'classification', dataset: 'MNIST',    description: '识别 MNIST 数据集中的 0-9 数字' },
    { id: 'cifar-classify',  name: '图像分类',       type: 'classification', dataset: 'CIFAR-10', description: '对 CIFAR-10 的 10 类图像分类' },
    { id: 'binary-classify', name: '二分类示例',     type: 'classification', dataset: 'Custom',   description: '两类别分类演示' },
    { id: 'regression',      name: '回归任务（占位）', type: 'regression',   dataset: 'Custom',   description: '回归任务（待后端支持）' }
  ];

  builtinTrainingDatasets: TrainingDatasetOption[] = [
    {
      id: 'mnist-1000',
      name: 'MNIST 全量',
      source: 'builtin',
      kind: 'image',
      description: '28x28 灰度手写数字，包含训练集和测试集共 70000 张。',
      sampleCount: 70000,
      classCount: 10,
      inputShape: '28 x 28 x 1',
      recommendedSplit: '70% / 15% / 15%',
      labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    },
    {
      id: 'cifar10-500',
      name: 'CIFAR-10 全量',
      source: 'builtin',
      kind: 'image',
      description: '32x32 RGB 彩色图片，覆盖 10 个常见物体类别，共 60000 张。',
      sampleCount: 60000,
      classCount: 10,
      inputShape: '32 x 32 x 3',
      recommendedSplit: '70% / 15% / 15%',
      labels: ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    },
    {
      id: 'cifar10-5000',
      name: 'CIFAR-10 5000 张',
      source: 'builtin',
      kind: 'image',
      description: '从 CIFAR-10 全量数据中按类别均衡抽取 5000 张图片。',
      sampleCount: 5000,
      classCount: 10,
      inputShape: '32 x 32 x 3',
      recommendedSplit: '70% / 15% / 15%',
      labels: ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    },
    {
      id: 'iris',
      name: '鸢尾花数据集',
      source: 'builtin',
      kind: 'table',
      description: '4 维表格特征，适合全连接网络分类演示。',
      sampleCount: 150,
      classCount: 3,
      inputShape: '4 numeric features',
      recommendedSplit: '80% / 20%',
      labels: ['setosa', 'versicolor', 'virginica']
    },
    {
      id: 'points-2d',
      name: '二维分类数据集',
      source: 'builtin',
      kind: 'points',
      description: '二维坐标点，适合展示决策边界和二分类过程。',
      sampleCount: 300,
      classCount: 2,
      inputShape: 'x, y',
      recommendedSplit: '70% / 15% / 15%',
      labels: ['class A', 'class B']
    }
  ];

  selectedTrainingDatasetId = 'mnist-1000';
  trainingDatasetDetail: TrainingDatasetDetail | null = null;
  trainingDatasetError = '';
  trainingDatasetLoading = false;
  trainingBackendNotice = '正在连接 Spring Boot 后端...';
  datasetImportDraft: DatasetImportDraft = {
    status: 'idle',
    message: '尚未导入自定义数据。',
    files: [],
    detectedKind: null,
    detail: null,
    csvHeaders: [],
    selectedLabelColumn: '',
    selectedClassCount: null
  };

  private subs = new Subscription();
  private tensorPreviewCache = new WeakMap<ForwardTensor, { mode: 'rgb' | 'gray'; width: number; height: number; colors?: string[]; values?: number[] }>();
  private tensorImagePreviewCache = new WeakMap<ForwardTensor, string>();
  private channelImagePreviewCache = new WeakMap<ChannelPreviewItem, string>();
  private rgbColorsCache = new WeakMap<object, string[]>();
  private tensorChannelPreviewCache = new WeakMap<ForwardTensor, ChannelPreviewItem[]>();
  private forwardDebounceTimer: number | null = null;
  private forwardRequestSeq = 0;
  private forwardInFlight = false;
  private forwardRerunRequested = false;

  constructor(
    private route: ActivatedRoute,
    private trainingSvc: TrainingRuntimeService,
    private trainingDatasetApi: TrainingDatasetApiService,
    private forwardBackend: ForwardBackendService,
    private authSvc: AuthService,
    private forwardRecordSvc: ForwardRecordService
  ) {}

  ngOnInit(): void {
    this.applyTemplate();
    this.subs.add(this.route.data.subscribe(data => {
      const routedMode = data['mode'] as AppMode | undefined;
      if (routedMode && routedMode !== this.mode) {
        this.setMode(routedMode);
      }
    }));
    this.loadLocalImageSamples();
    void this.loadTrainingDatasets();
    this.subs.add(this.trainingSvc.state$.subscribe(s => {
      this.trainingStatus  = s.status;
      this.trainingEpoch   = s.currentEpoch;
      this.trainingLr      = s.currentLr;
      this.trainingLoss    = s.latestLoss;
      this.trainingValLoss = s.latestValLoss;
      this.trainingAcc     = s.latestAccuracy;
      this.trainingValAcc  = s.latestValAccuracy;
      this.trainingGradientNorm = s.latestGradientNorm;
      this.trainingWeightMean = s.latestWeightMean;
      this.trainingWeightStd = s.latestWeightStd;
      this.trainingElapsedSeconds = s.elapsedSeconds;
      this.trainingEtaSeconds = s.etaSeconds;
      this.trainingCurrentBatchValue = s.currentBatch ?? 0;
      this.trainingTotalBatchesValue = s.totalBatches ?? 0;
    }));
    this.subs.add(this.trainingSvc.history$.subscribe(h => this.trainingHistory = h));
    this.subs.add(this.trainingSvc.logs$.subscribe(l => this.trainingLogs = l));
    this.subs.add(this.trainingSvc.testResult$.subscribe(result => {
      this.trainingTestResult = result;
      if (result && this.authUser) void this.loadTrainingCheckpoints();
    }));
    this.subs.add(this.authSvc.user$.subscribe(user => {
      this.authUser = user;
      if (user && this.showRecordDrawer) {
        this.loadForwardRecords();
      }
      if (user) {
        void this.loadTrainingCheckpoints();
      } else {
        this.trainingCheckpoints = [];
        this.selectedCheckpointId = null;
      }
    }));
    this.authSvc.restoreSession();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    if (this.forwardDebounceTimer !== null) {
      window.clearTimeout(this.forwardDebounceTimer);
      this.forwardDebounceTimer = null;
    }
    void this.trainingSvc.pause();
  }

  // ── Getters ──────────────────────────────────────────
  get layerCount() { return this.layers.length; }
  get parameterCount() { return SimEngine.parameterCount(this.layers, this.connections); }
  get layerPalette(): LayerType[] { return ['conv2d', 'residual', 'pool2d', 'flatten', 'dense', 'activation', 'dropout']; }
  get selectedTemplate() { return this.modelTemplates.find(t => t.id === this.selectedTemplateId); }
  get selectedLayer() { return this.layers.find(l => l.id === this.selectedLayerId); }
  get selectedCheckpoint() { return this.trainingCheckpoints.find(item => item.id === this.selectedCheckpointId) ?? null; }
  get inputLayer(): InputLayer | undefined { const l = this.layers.find(l => l.type === 'input'); return l?.type === 'input' ? l : undefined; }
  get outputLayer() { const l = this.layers.find(l => l.type === 'output'); return l?.type === 'output' ? l : undefined; }
  get datasetSamples() { return this.datasets[this.selectedDataset] ?? []; }
  get selectedSample() { return this.datasetSamples.find(s => s.id === this.selectedSampleId); }
  get localDatasetNames(): string[] {
    return [...new Set(this.localImageSamples.map(sample => sample.category))];
  }
  get activeLocalImageSamples(): LocalImageSample[] {
    return this.localImageSamples.filter(sample => sample.category === this.selectedDataset);
  }
  get selectedLocalImageSample(): LocalImageSample | undefined {
    return this.localImageSamples.find(sample => sample.id === this.selectedLocalImageId);
  }
  localDatasetSampleCount(dataset: string): number {
    return this.localImageSamples.filter(sample => sample.category === dataset).length;
  }
  get trainingDatasetReady(): boolean { return !!this.trainingDatasetDetail?.hasLabels; }
  get trainingDatasetMaxLabelCount(): number {
    return Math.max(1, ...(this.trainingDatasetDetail?.labelDistribution ?? []).map(i => i.count));
  }
  get importedDatasetSelected(): boolean {
    return !!this.datasetImportDraft.detail && this.selectedTrainingDatasetId === this.datasetImportDraft.detail.id;
  }
  get datasetSplitSumPercent(): number {
    const ds = this.trainingDatasetDetail;
    if (!ds) return 0;
    return Math.round((ds.trainRatio + ds.valRatio + ds.testRatio) * 1000) / 10;
  }
  get datasetSplitError(): string {
    const ds = this.trainingDatasetDetail;
    if (!ds) return '请先选择或导入一个训练数据集。';
    const ratios = [ds.trainRatio, ds.valRatio, ds.testRatio];
    if (ratios.some(v => !Number.isFinite(v) || v < 0 || v > 1)) return '划分比例必须在 0% 到 100% 之间。';
    if (ds.trainRatio <= 0) return '训练集比例必须大于 0%。';
    if (Math.abs(ds.trainRatio + ds.valRatio + ds.testRatio - 1) > 0.001) return '训练集、验证集、测试集比例总和必须等于 100%。';
    return '';
  }
  get hasTrainingModelError(): boolean {
    return this.trainingModelIssues.some(issue => issue.level === 'error');
  }
  get trainingModelIssues(): Array<{ level: 'ok' | 'warn' | 'error'; message: string }> {
    const issues: Array<{ level: 'ok' | 'warn' | 'error'; message: string }> = [];
    const ds = this.trainingDatasetDetail;
    if (!ds) return [{ level: 'error', message: '请先选择训练数据集。' }];
    if (!this.inputLayer) issues.push({ level: 'error', message: '网络缺少输入层。' });
    if (!this.outputLayer) issues.push({ level: 'error', message: '网络缺少输出层。' });

    if (this.outputLayer && ds.classCount > 0 && this.outputLayer.params.units !== ds.classCount) {
      issues.push({
        level: 'error',
        message: `输出层类别数为 ${this.outputLayer.params.units}，当前数据集需要 ${ds.classCount}。`
      });
    }

    if (this.inputLayer && ds.kind === 'image') {
      if (this.inputLayer.params.inputKind === 'table') {
        issues.push({ level: 'error', message: '图像数据集需要使用图像输入层，请把输入类型改为“图像输入”。' });
      }
      const shape = ds.inputShape.match(/(\d+)\s*x\s*(\d+)\s*x\s*(\d+)/i);
      if (shape) {
        const [, h, w, c] = shape.map(Number);
        const p = this.inputLayer.params;
        if (p.height !== h || p.width !== w || p.channels !== c) {
          issues.push({ level: 'warn', message: `输入层为 ${p.height}x${p.width}x${p.channels}，数据集为 ${h}x${w}x${c}。` });
        }
      }
    }

    if (ds.kind === 'table' || ds.kind === 'points') {
      if (this.inputLayer?.params.inputKind !== 'table') {
        issues.push({ level: 'error', message: 'CSV/表格数据是向量输入，请把输入层类型改为“CSV 向量输入”，或选择 CSV / Tabular MLP 模板。' });
      }
      const imageOnlyLayers = this.layers.filter(layer => ['conv2d', 'pool2d', 'residual'].includes(layer.type));
      if (imageOnlyLayers.length > 0) {
        issues.push({ level: 'error', message: 'CSV/表格数据不能直接使用 Conv2D、池化或残差块，请改用 Dense / Activation / Dropout / Output。' });
      }
      if (this.layers.some(layer => layer.type === 'flatten')) {
        issues.push({ level: 'warn', message: 'CSV/表格数据已经是向量，Flatten 通常不需要。' });
      }
      issues.push({ level: 'warn', message: 'CSV/表格数据训练时会由后端进行数值/类别特征编码，并作为向量输入。' });
    }

    if (!this.layers.some(l => l.type === 'dense' || l.type === 'conv2d')) {
      issues.push({ level: 'error', message: '网络至少需要一个可训练层。' });
    }

    for (const issue of this.trainingValidationIssues) {
      issues.push({
        level: issue.severity === 'error' ? 'error' : 'warn',
        message: `${issue.layerName}: ${issue.message}`
      });
    }

    return issues.length ? issues : [{ level: 'ok', message: '当前网络结构可用于训练配置。' }];
  }

  get trainingValidationIssues(): LayerValidationIssue[] {
    return this.analyzeTrainingNetwork().issues;
  }

  get trainingLayerShapeMap(): Record<number, string> {
    return this.analyzeTrainingNetwork().shapeMap;
  }

  get selectedForwardResult(): ForwardLayerResult | null {
    if (!this.forwardResult?.layerResults.length) return null;
    return this.forwardResult.layerResults.find(r => r.layerId === this.selectedLayerId)
      ?? this.forwardResult.layerResults[0];
  }

  get selectedBars(): number[] {
    return this.normBars((this.selectedForwardResult?.visualization.values ?? []).slice(0, 64));
  }

  get inputColorModeOptions(): Array<{ value: 'original' | 'rgb' | 'grayscale'; label: string }> {
    const channels = this.currentInputAsset?.originalChannels ?? this.inputLayer?.params.channels ?? 1;
    if (channels === 1) {
      return [
        { value: 'original', label: '原始（灰度）' },
        { value: 'grayscale', label: '灰度（单通道）' }
      ];
    }
    if (channels >= 3) {
      return [
        { value: 'original', label: '原始（保持输入通道）' },
        { value: 'rgb', label: 'RGB（三通道）' },
        { value: 'grayscale', label: '灰度（单通道）' }
      ];
    }
    return [{ value: 'original', label: '原始' }];
  }

  get selectedConvLayer() {
    const layer = this.selectedLayer;
    return layer?.type === 'conv2d' ? layer : null;
  }

  get selectedConvInChannels(): number {
    const layer = this.selectedConvLayer;
    if (!layer) return 1;
    const result = this.forwardResult?.layerResults.find(r => r.layerId === layer.id);
    const shape = result?.inputShapes?.[0];
    if (shape && shape.length === 3) return Math.max(1, shape[2]);
    return Math.max(1, this.inputLayer?.params.channels ?? 1);
  }

  get convOutChannelIndices(): number[] {
    const out = this.selectedConvLayer?.params.outChannels ?? 1;
    return Array.from({ length: Math.max(1, out) }, (_, i) => i);
  }

  get convInChannelIndices(): number[] {
    const inC = this.selectedConvInChannels;
    return Array.from({ length: Math.max(1, inC) }, (_, i) => i);
  }

  get showConvInChannelSelector(): boolean {
    return this.selectedConvInChannels > 1;
  }

  get editableKernelMatrix(): number[][] {
    const layer = this.selectedConvLayer;
    if (!layer) return [];
    this.ensureConvKernelBank(layer);
    return layer.params.kernels?.[this.selectedKernelOutChannel]?.weights?.[this.selectedKernelInChannel]
      ?? layer.params.kernelMatrix
      ?? [];
  }

  get finalTensorMode(): 'image' | 'vector' | 'none' {
    const shapeLen = this.forwardResult?.finalTensor?.shape.length ?? 0;
    if (shapeLen === 3) return 'image';
    if (shapeLen >= 1) return 'vector';
    return 'none';
  }

  get finalBars(): number[] {
    if (this.finalTensorMode !== 'vector') return [];
    return this.normBars((this.forwardResult?.finalTensor?.values ?? []).slice(0, 32));
  }

  get finalImageViz() {
    const t = this.forwardResult?.finalTensor;
    if (!t || t.shape.length !== 3) return null;
    const previewTensor = this.sampleTensorForPreview(t, MAX_PREVIEW_GRID_SIDE);
    const [h, w, c] = previewTensor.shape as [number, number, number];
    const srcValues = previewTensor.values;
    const channelPreviews = this.buildChannelPreviews(t, 4);
    if (c === 3 && (t.colorMode === 'rgb' || t.colorMode === undefined)) {
      const colors = Array.from({ length: h * w }, (_, i) => {
        const base = i * 3;
        return `rgb(${Math.round((srcValues[base] ?? 0) * 255)},${Math.round((srcValues[base + 1] ?? 0) * 255)},${Math.round((srcValues[base + 2] ?? 0) * 255)})`;
      });
      return { mode: 'rgb' as const, colors, width: w, height: h, channels: c, channelPreviews };
    }

    return {
      mode: 'gray' as const,
      values: channelPreviews[0]?.values ?? [],
      width: w,
      height: h,
      channels: c,
      channelPreviews
    };
  }

  get selectedChannelPreviews(): ChannelPreviewItem[] {
    return (this.selectedForwardResult?.visualization.channelPreviews ?? []).slice(0, 4);
  }

  get selectedChannelCount(): number {
    const tensor = this.selectedForwardResult?.tensor;
    return tensor && tensor.shape.length === 3 ? tensor.shape[2] : 0;
  }

  get finalChannelCount(): number {
    const tensor = this.forwardResult?.finalTensor;
    return tensor && tensor.shape.length === 3 ? tensor.shape[2] : 0;
  }

  /** 当前选中层可视化是否为 RGB（channels=3）*/
  get selectedIsRgb(): boolean {
    const viz = this.selectedForwardResult?.visualization;
    return viz?.mode === 'image' && (viz.channels ?? 1) === 3;
  }

  /** 选中层 RGB 颜色数组（用于 RGB 图像渲染）*/
  get selectedRgbColors(): string[] {
    const viz = this.selectedForwardResult?.visualization;
    if (!viz || viz.mode !== 'image' || (viz.channels ?? 1) !== 3) return [];
    const cached = this.rgbColorsCache.get(viz as unknown as object);
    if (cached) return cached;
    const tensor = this.selectedForwardResult?.tensor;
    if (tensor?.shape.length === 3) {
      const previewTensor = this.sampleTensorForPreview(tensor, MAX_PREVIEW_GRID_SIDE);
      const vals = previewTensor.values;
      const [h, w] = previewTensor.shape as [number, number, number];
      const n = h * w;
      const colors = Array.from({ length: n }, (_, i) => {
        const base = i * 3;
        return `rgb(${Math.round((vals[base]??0)*255)},${Math.round((vals[base+1]??0)*255)},${Math.round((vals[base+2]??0)*255)})`;
      });
      this.rgbColorsCache.set(viz as unknown as object, colors);
      return colors;
    }
    const vals = viz.values;
    const n = Math.min((viz.width ?? 1) * (viz.height ?? 1), MAX_PREVIEW_GRID_SIDE * MAX_PREVIEW_GRID_SIDE);
    const colors = Array.from({ length: n }, (_, i) => {
      const base = i * 3;
      return `rgb(${Math.round((vals[base]??0)*255)},${Math.round((vals[base+1]??0)*255)},${Math.round((vals[base+2]??0)*255)})`;
    });
    this.rgbColorsCache.set(viz as unknown as object, colors);
    return colors;
  }

  /** 原始输入预览（RGB 或灰度）*/
  get originalInputPreview() {
    const t = this.currentInputAsset?.originalTensor;
    if (!t || t.shape.length !== 3) return null;
    return this.previewTensorForGrid(t);
  }

  /** 预处理后预览（RGB 或灰度）*/
  get preparedInputPreview() {
    const t = this.currentInputAsset?.prepared.displayTensor;
    if (!t || t.shape.length !== 3) return null;
    return this.previewTensorForGrid(t);
  }

  get originalInputImageUrl(): string {
    return this.currentInputAsset?.previewUrl
      || this.tensorToImageDataUrl(this.currentInputAsset?.originalTensor ?? null);
  }

  get preparedInputImageUrl(): string {
    return this.tensorToImageDataUrl(this.currentInputAsset?.prepared.displayTensor ?? null);
  }

  get selectedTensorImageUrl(): string {
    const tensor = this.selectedForwardResult?.tensor ?? null;
    return this.tensorToImageDataUrl(tensor, !(tensor?.shape.length === 3 && tensor.shape[2] === 3 && tensor.colorMode === 'rgb'));
  }

  get finalTensorImageUrl(): string {
    const tensor = this.forwardResult?.finalTensor ?? null;
    return this.tensorToImageDataUrl(tensor, !(tensor?.shape.length === 3 && tensor.shape[2] === 3 && tensor.colorMode === 'rgb'));
  }

  channelPreviewImageUrl(channel: ChannelPreviewItem): string {
    const cached = this.channelImagePreviewCache.get(channel);
    if (cached) return cached;
    const url = this.grayValuesToImageDataUrl(channel.values, channel.width, channel.height);
    this.channelImagePreviewCache.set(channel, url);
    return url;
  }

  openImageViewer(title: string, url: string, meta = ''): void {
    if (!url) return;
    this.imageViewer = { open: true, title, url, meta };
  }

  closeImageViewer(): void {
    this.imageViewer = { open: false, title: '', url: '', meta: '' };
  }

  get isRgbInput(): boolean { return (this.currentInputAsset?.originalChannels ?? 1) >= 3; }
  get lossPolyline() { return this.metricPolyline('loss', this.lossChartDomain); }
  get valLossPolyline() { return this.metricPolyline('valLoss', this.lossChartDomain); }
  get accPolyline()  { return this.metricPolyline('accuracy', this.accuracyChartDomain); }
  get valPolyline()  { return this.metricPolyline('valAccuracy', this.accuracyChartDomain); }
  get lrPolyline() { return this.metricPolyline('lr', this.lrChartDomain); }
  get gradientPolyline() { return this.metricPolyline('gradientNorm', this.gradientChartDomain); }
  get lossAxisTicks() { return this.chartAxisTicks(this.lossChartDomain, 'number'); }
  get accuracyAxisTicks() { return this.chartAxisTicks(this.accuracyChartDomain, 'percent'); }
  get lrAxisTicks() { return this.chartAxisTicks(this.lrChartDomain, 'lr'); }
  get gradientAxisTicks() { return this.chartAxisTicks(this.gradientChartDomain, 'number'); }
  private get lossChartDomain(): [number, number] {
    return this.chartDomain(['loss', 'valLoss'], { min: 0, padRatio: 0.08, fallbackMax: 1 });
  }
  private get accuracyChartDomain(): [number, number] {
    return [0, 1];
  }
  private get lrChartDomain(): [number, number] {
    return this.chartDomain(['lr'], { min: 0, padRatio: 0.08, fallbackMax: Math.max(0.001, this.trainingConfig.learningRate) });
  }
  private get gradientChartDomain(): [number, number] {
    return this.chartDomain(['gradientNorm'], { min: 0, padRatio: 0.12, fallbackMax: 1 });
  }
  get trainingTotalBatches(): number {
    if (this.trainingTotalBatchesValue > 0) return this.trainingTotalBatchesValue;
    const ds = this.trainingDatasetDetail;
    if (!ds) return 0;
    const trainSamples = Math.max(1, Math.round(ds.sampleCount * ds.trainRatio));
    return Math.max(1, Math.ceil(trainSamples / Math.max(1, this.trainingConfig.batchSize)));
  }
  get trainingCurrentBatch(): number {
    if (this.trainingCurrentBatchValue > 0) return this.trainingCurrentBatchValue;
    if (this.trainingStatus === 'idle' || this.trainingEpoch === 0) return 0;
    return this.trainingTotalBatches;
  }
  get trainingProgressPercent(): number {
    return this.trainingConfig.totalEpochs > 0 ? (this.trainingEpoch / this.trainingConfig.totalEpochs) * 100 : 0;
  }
  get gradientAlert(): string {
    if (this.trainingGradientNorm < 0.02) return '梯度可能消失';
    if (this.trainingGradientNorm > 2.5) return '梯度可能爆炸';
    return '梯度稳定';
  }
  get weightHistogramBins(): Array<{ label: string; value: number }> {
    const mean = this.trainingWeightMean;
    const std = Math.max(0.01, this.trainingWeightStd);
    return Array.from({ length: 13 }, (_, i) => {
      const x = -3 + i * 0.5;
      const density = Math.exp(-0.5 * Math.pow((x * std - mean) / std, 2));
      return { label: (x * std).toFixed(2), value: density };
    });
  }
  get maxWeightBin(): number {
    return Math.max(1e-6, ...this.weightHistogramBins.map(bin => bin.value));
  }

  private metricPolyline(metric: TrainingChartMetric, domain: [number, number]): string {
    if (this.trainingHistory.length === 0) return '';
    const maxStep = Math.max(1, ...this.trainingHistory.map(point => point.step));
    const [minValue, maxValue] = domain;
    const span = Math.max(0.000001, maxValue - minValue);
    return this.trainingHistory
      .map(point => {
        const x = (point.step / maxStep) * 100;
        const y = 100 - ((point[metric] - minValue) / span) * 100;
        return `${x.toFixed(2)},${Math.min(100, Math.max(0, y)).toFixed(2)}`;
      })
      .join(' ');
  }

  private chartDomain(
    metrics: TrainingChartMetric[],
    options: { min?: number; max?: number; padRatio?: number; fallbackMax?: number } = {}
  ): [number, number] {
    const values = this.trainingHistory
      .flatMap(point => metrics.map(metric => point[metric]))
      .filter(value => Number.isFinite(value));
    const rawMin = values.length ? Math.min(...values) : 0;
    const rawMax = values.length ? Math.max(...values) : (options.fallbackMax ?? 1);
    const min = options.min ?? rawMin;
    let max = options.max ?? Math.max(rawMax, options.fallbackMax ?? rawMax);
    if (max <= min) max = min + Math.max(0.001, Math.abs(min) * 0.1 || 1);
    max += (max - min) * (options.padRatio ?? 0);
    return [min, max];
  }

  private chartAxisTicks(domain: [number, number], format: TrainingChartFormat): string[] {
    const [min, max] = domain;
    return [max, (max + min) / 2, min].map(value => this.formatChartTick(value, format));
  }

  private formatChartTick(value: number, format: TrainingChartFormat): string {
    if (format === 'percent') return `${Math.round(value * 100)}%`;
    if (format === 'lr') {
      if (value === 0) return '0';
      return value < 0.001 ? value.toExponential(1) : value.toFixed(4).replace(/0+$/, '').replace(/\.$/, '');
    }
    if (value >= 10) return value.toFixed(1);
    if (value >= 1) return value.toFixed(2);
    return value.toFixed(3);
  }

  get validationIssues(): LayerValidationIssue[] {
    return this.mode === 'training' ? this.trainingValidationIssues : (this.forwardResult?.validationIssues ?? []);
  }

  get fieldIssueMap(): Record<number, Record<string, string[]>> {
    const map: Record<number, Record<string, string[]>> = {};
    for (const issue of this.validationIssues) {
      if (!issue.field) continue;
      map[issue.layerId] ??= {};
      map[issue.layerId][issue.field] = [...(map[issue.layerId][issue.field] ?? []), issue.message];
    }
    return map;
  }

  get errorLayerIdList(): number[] {
    const ids = new Set(this.validationIssues.filter(i => i.severity === 'error').map(i => i.layerId));
    if (this.mode !== 'training') {
      for (const err of this.forwardResult?.errors ?? []) {
        const layerName = err.split(':')[0]?.trim();
        const layer = this.layers.find(l => l.name === layerName);
        if (layer) ids.add(layer.id);
      }
    }
    return [...ids];
  }

  get layerErrors(): Record<number, string[]> {
    const map: Record<number, string[]> = {};
    for (const issue of this.validationIssues.filter(i => i.severity === 'error')) {
      map[issue.layerId] = [...(map[issue.layerId] ?? []), issue.message];
    }
    return map;
  }

  hasLayerError(id: number): boolean { return !!(this.layerErrors[id]?.length); }
  hasFieldError(layerId: number, field: string): boolean { return !!(this.fieldIssueMap[layerId]?.[field]?.length); }
  fieldErrorText(layerId: number, field: string): string { return this.fieldIssueMap[layerId]?.[field]?.[0] ?? ''; }
  get globalErrorMessages(): string[] { return this.forwardResult?.errors ?? []; }

  get authTitle(): string { return this.authMode === 'login' ? '登录' : '注册'; }
  get canSaveForwardRecord(): boolean {
    return this.mode === 'forward' && !!this.forwardResult && !this.forwardBusy && !this.pendingForwardChanges;
  }

  openAuthModal(mode: 'login' | 'register' = 'login'): void {
    this.authMode = mode;
    this.authError = '';
    this.authDraft = { username: '', password: '', displayName: '' };
    this.showAuthModal = true;
  }

  closeAuthModal(): void {
    if (this.authBusy) return;
    this.showAuthModal = false;
    this.authError = '';
  }

  async submitAuth(): Promise<void> {
    if (this.authBusy) return;
    this.authBusy = true;
    this.authError = '';
    try {
      if (this.authMode === 'login') {
        await this.authSvc.login(this.authDraft.username, this.authDraft.password);
      } else {
        await this.authSvc.register(this.authDraft.username, this.authDraft.password, this.authDraft.displayName);
      }
      this.showAuthModal = false;
      this.recordSuccess = `${this.authTitle}成功`;
    } catch (err) {
      this.authError = err instanceof Error ? err.message : '认证请求失败';
    } finally {
      this.authBusy = false;
    }
  }

  logout(): void {
    this.authSvc.logout();
    this.forwardRecords = [];
    this.showRecordDrawer = false;
    this.recordSuccess = '已退出登录';
  }

  openSaveRecordModal(): void {
    this.recordError = '';
    this.recordSuccess = '';
    if (!this.authUser) {
      this.openAuthModal('login');
      this.recordError = '请先登录后再保存 A 模式记录';
      return;
    }
    if (!this.canSaveForwardRecord) {
      this.recordError = '请先手动点击“开始计算”，完成后再保存记录';
      return;
    }
    const now = new Date();
    this.recordNameDraft = `A模式记录 ${now.getFullYear()}-${now.getMonth() + 1}-${now.getDate()} ${now.getHours()}:${String(now.getMinutes()).padStart(2, '0')}`;
    this.showSaveRecordModal = true;
  }

  closeSaveRecordModal(): void {
    if (this.recordBusy) return;
    this.showSaveRecordModal = false;
    this.recordError = '';
  }

  async saveForwardRecord(): Promise<void> {
    if (!this.authUser || !this.canSaveForwardRecord || this.recordBusy) return;
    const name = this.recordNameDraft.trim();
    if (!name) {
      this.recordError = '请给这次记录命名';
      return;
    }

    this.recordBusy = true;
    this.recordError = '';
    try {
      const record = await this.forwardRecordSvc.create({
        name,
        templateId: this.selectedTemplateId,
        datasetName: this.selectedDataset,
        layerCount: this.layerCount,
        parameterCount: this.parameterCount,
        previewImageDataUrl: this.currentInputPreviewDataUrl(),
        snapshot: this.buildForwardRecordSnapshot()
      });
      this.showSaveRecordModal = false;
      this.recordSuccess = `已保存：${record.name}`;
      await this.loadForwardRecords();
      this.showRecordDrawer = true;
    } catch (err) {
      this.recordError = err instanceof Error ? err.message : '保存记录失败';
    } finally {
      this.recordBusy = false;
    }
  }

  async toggleRecordDrawer(): Promise<void> {
    this.recordError = '';
    this.recordSuccess = '';
    if (!this.authUser) {
      this.openAuthModal('login');
      this.recordError = '请先登录后再查看历史记录';
      return;
    }
    this.showRecordDrawer = !this.showRecordDrawer;
    if (this.showRecordDrawer) {
      await this.loadForwardRecords();
    }
  }

  async loadForwardRecords(): Promise<void> {
    if (!this.authUser) return;
    this.recordBusy = true;
    this.recordError = '';
    try {
      this.forwardRecords = await this.forwardRecordSvc.list();
    } catch (err) {
      this.recordError = err instanceof Error ? err.message : '读取历史记录失败';
    } finally {
      this.recordBusy = false;
    }
  }

  async restoreForwardRecord(recordId: number): Promise<void> {
    if (this.recordBusy) return;
    this.recordBusy = true;
    this.recordError = '';
    try {
      const detail = await this.forwardRecordSvc.detail(recordId);
      await this.applyForwardRecord(detail);
      this.recordSuccess = `已回溯：${detail.name}`;
      this.showRecordDrawer = false;
    } catch (err) {
      this.recordError = err instanceof Error ? err.message : '回溯记录失败';
    } finally {
      this.recordBusy = false;
    }
  }

  async deleteForwardRecord(recordId: number): Promise<void> {
    if (this.recordBusy) return;
    this.recordBusy = true;
    this.recordError = '';
    try {
      await this.forwardRecordSvc.delete(recordId);
      this.forwardRecords = this.forwardRecords.filter(record => record.id !== recordId);
      this.recordSuccess = '记录已删除';
    } catch (err) {
      this.recordError = err instanceof Error ? err.message : '删除记录失败';
    } finally {
      this.recordBusy = false;
    }
  }

  recordImageUrl(record: ForwardRecordSummary): string {
    return this.forwardRecordSvc.imageUrl(record.imagePath);
  }

  openSelectedChannelsModal(): void {
    const tensor = this.selectedForwardResult?.tensor;
    if (!tensor || tensor.shape.length !== 3) return;
    this.channelModalTitle = `${this.selectedForwardResult?.layerName ?? '当前层'} · 全部通道 (${tensor.shape[2]})`;
    this.channelModalPreviews = this.buildChannelPreviews(tensor);
    this.showChannelModal = true;
  }

  openFinalChannelsModal(): void {
    const tensor = this.forwardResult?.finalTensor;
    if (!tensor || tensor.shape.length !== 3) return;
    this.channelModalTitle = `最终输出 · 全部通道 (${tensor.shape[2]})`;
    this.channelModalPreviews = this.buildChannelPreviews(tensor);
    this.showChannelModal = true;
  }

  closeChannelModal(): void {
    this.showChannelModal = false;
    this.channelModalPreviews = [];
    this.channelModalTitle = '';
  }

  // ── Mode ─────────────────────────────────────────────
  setMode(m: AppMode): void {
    this.mode = m;
    if (m === 'forward') { void this.trainingSvc.pause(); this.runForward(); }
  }

  // ── Template ─────────────────────────────────────────
  applyTemplate(): void {
    const tpl = this.selectedTemplate;
    if (!tpl) return;
    this.layers = tpl.layers.map((d, i) => ({
      ...d, id: i + 1, inputs: i === 0 ? [] : [i], params: structuredClone(d.params)
    } as NetworkLayer));
    this.nextLayerId = this.layers.length + 1;
    this.selectedLayerId = this.layers[1]?.id ?? this.layers[0]?.id ?? -1;
    this.syncTemplateWithTrainingDataset();
    this.rebuildTopology();
    this.rebuildInputAsset();
    this.runForward();
  }

  // ── Layer editing ─────────────────────────────────────
  addLayer(type: LayerType): void {
    if (type === 'input' || type === 'output') return;
    const layer = this.defaultLayer(type, this.nextLayerId++);
    const outIdx = this.layers.findIndex(l => l.type === 'output');
    this.layers.splice(outIdx < 0 ? this.layers.length : outIdx, 0, layer);
    this.selectedLayerId = layer.id;
    this.rebuildTopology(); this.runForward();
  }

  removeSelectedLayer(): void {
    const t = this.selectedLayer;
    if (!t || t.type === 'input' || t.type === 'output') return;
    this.layers = this.layers.filter(l => l.id !== t.id);
    this.selectedLayerId = this.layers[1]?.id ?? this.layers[0]?.id ?? -1;
    this.rebuildTopology(); this.runForward();
  }

  moveSelectedLayer(dir: 'left' | 'right'): void {
    const idx = this.layers.findIndex(l => l.id === this.selectedLayerId);
    if (idx < 0) return;
    const ti = dir === 'left' ? idx - 1 : idx + 1;
    if (ti <= 0 || ti >= this.layers.length - 1) return;
    const arr = [...this.layers];
    arr.splice(ti, 0, arr.splice(idx, 1)[0]);
    this.layers = arr;
    this.rebuildTopology(); this.runForward();
  }

  onLayerPicked(id: number): void { this.selectedLayerId = id; }

  /** 拖拽重排序 */
  onLayersReordered(newLayers: NetworkLayer[]): void {
    this.layers = newLayers;
    this.rebuildTopology();
    this.runForward();
  }

  /** 从 palette 拖拽插入新层 */
  onNewLayerDropped(event: { type: string; index: number }): void {
    const type = event.type as LayerType;
    if (type === 'input' || type === 'output') return;
    const layer = this.defaultLayer(type, this.nextLayerId++);
    // 插入到指定位置（但不能插到 input 前或 output 后）
    const safeIndex = Math.max(1, Math.min(event.index, this.layers.length - 1));
    this.layers.splice(safeIndex, 0, layer);
    this.selectedLayerId = layer.id;
    this.rebuildTopology();
    this.runForward();
  }

  onLayerConfigChange(): void {
    this.syncConvKernelSelectors();
    this.syncKernelShape();
    this.rebuildInputAsset();
    this.runForward();
  }

  onKernelSizeChange(): void { this.syncKernelShape(); this.runForward(); }

  onKernelCellInput(r: number, c: number, v: string): void {
    const l = this.selectedLayer;
    if (!l || l.type !== 'conv2d') return;
    this.ensureConvKernelBank(l);
    const matrix = l.params.kernels?.[this.selectedKernelOutChannel]?.weights?.[this.selectedKernelInChannel];
    if (!matrix) return;
    matrix[r][c] = Number.isFinite(+v) ? +v : 0;
    l.params.kernelMatrix = l.params.kernels?.[0]?.weights?.[0]?.map(row => [...row]) ?? l.params.kernelMatrix;
    this.runForward();
  }

  onKernelChannelChange(): void {
    this.syncConvKernelSelectors();
    const l = this.selectedConvLayer;
    if (!l) return;
    this.ensureConvKernelBank(l);
  }

  applyKernelPreset(preset: KernelPreset): void {
    const l = this.selectedLayer;
    if (!l || l.type !== 'conv2d') return;
    l.params.kernelSize = 3;
    this.ensureConvKernelBank(l);
    const matrix = l.params.kernels?.[this.selectedKernelOutChannel]?.weights?.[this.selectedKernelInChannel];
    if (matrix) {
      for (let y = 0; y < 3; y += 1) {
        for (let x = 0; x < 3; x += 1) {
          matrix[y][x] = preset.matrix[y][x];
        }
      }
    }
    l.params.kernelMatrix = l.params.kernels?.[0]?.weights?.[0]?.map(row => [...row]) ?? preset.matrix.map(row => [...row]);
    this.syncKernelShape();
    this.runForward();
  }

  // ── Dataset / Input ───────────────────────────────────
  selectDataset(name: string): void {
    this.selectedDataset = name;
    this.selectedSampleId = 1;
    this.showSamplePicker = false;
    const firstLocal = this.activeLocalImageSamples[0];
    if (firstLocal) {
      this.chooseLocalImageSample(firstLocal);
    } else {
      this.clearLocalImageSelection();
      this.rebuildInputAsset();
      this.runForward();
    }
  }

  chooseSample(id: number): void {
    this.selectedSampleId = id;
    this.uploadedImageUrl = ''; this.uploadedImageData = null; this.uploadError = '';
    this.clearLocalImageSelection();
    this.showSamplePicker = false;
    this.rebuildInputAsset(); this.runForward();
  }

  toggleSamplePicker(): void { this.showSamplePicker = !this.showSamplePicker; }
  closeSamplePicker(): void  { this.showSamplePicker = false; }

  onImageUpload(e: Event): void {
    const input = e.target as HTMLInputElement;
    const file = input.files?.[0];
    // 重置 input，允许重复上传同一文件
    input.value = '';
    if (!file) return;

    // 文件类型校验
    if (!file.type.startsWith('image/')) {
      this.uploadError = `不支持的文件类型：${file.type}`;
      return;
    }
    // 文件大小限制 30MB（上传后会自动按最大边缩放）
    if (file.size > 30 * 1024 * 1024) {
      this.uploadError = '图片文件过大（>30MB），请换一张更小的图片';
      return;
    }

    this.uploadError = '';
    this.clearLocalImageSelection();
    const reader = new FileReader();
    reader.onerror = () => { this.uploadError = '文件读取失败，请重试'; };
    reader.onload = () => {
      const url = typeof reader.result === 'string' ? reader.result : '';
      if (!url) { this.uploadError = '图片读取结果为空'; return; }
      this.decodeAndResizeImage(url).then(({ imageData, previewUrl }) => {
        this.uploadedImageUrl = previewUrl;
        this.uploadedImageData = imageData;
        this.applyUploadComputeProfile(imageData.width, imageData.height);
        this.uploadError = '';
        this.rebuildInputAsset();
        this.runForward();
      }).catch(err => {
        this.uploadError = `图片处理失败：${err?.message ?? '未知错误'}`;
      });
    };
    reader.readAsDataURL(file);
  }

  async chooseLocalImageSample(sample: LocalImageSample): Promise<void> {
    this.localImageError = '';
    try {
      const { imageData, previewUrl } = await this.decodeAndResizeImage(sample.url);
      this.selectedLocalImageId = sample.id;
      this.localImageData = imageData;
      this.localImagePreviewUrl = previewUrl;
      this.uploadedImageUrl = '';
      this.uploadedImageData = null;
      this.uploadError = '';
      this.showSamplePicker = false;
      this.rebuildInputAsset();
      this.runForward();
    } catch (err) {
      this.localImageError = err instanceof Error ? err.message : '示例图片加载失败';
    }
  }

  // ── Forward pass ──────────────────────────────────────
  runForward(force = false): void {
    if (this.mode !== 'forward') return;
    if (!force && !this.autoForwardCompute) {
      this.pendingForwardChanges = true;
      this.forwardStatusMessage = '参数已更新，点击“开始计算”执行前向传播。';
      return;
    }
    if (this.forwardInFlight) {
      this.pendingForwardChanges = true;
      this.forwardRerunRequested = true;
      this.forwardStatusMessage = '计算中...';
      return;
    }
    const activeSeq = ++this.forwardRequestSeq;
    if (this.forwardDebounceTimer !== null) {
      window.clearTimeout(this.forwardDebounceTimer);
    }

    this.forwardDebounceTimer = window.setTimeout(async () => {
      const inputTensor = this.currentInputAsset?.prepared.tensor;
      if (!inputTensor) {
        this.forwardBackendError = '';
        this.forwardStatusMessage = 'No input asset available.';
        this.forwardResult = null;
        this.forwardLayerShapeMap = {};
        return;
      }

      this.forwardBusy = true;
      this.forwardInFlight = true;
      this.pendingForwardChanges = false;
      this.forwardStatusMessage = '计算中...';
      try {
        const remote = await this.forwardBackend.executeForward({
          layers: this.layers,
          connections: this.connections,
          inputTensor
        });
        this.forwardBackendError = '';
        this.forwardStatusMessage = '计算完成。';
        if (!this.forwardRerunRequested) {
          this.applyForwardResult(remote, activeSeq);
        }
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          this.forwardStatusMessage = '计算已取消。';
          return;
        }
        this.forwardBackendError = '后端不可用。';
        this.forwardStatusMessage = '后端请求失败。';
      } finally {
        this.forwardInFlight = false;
        const shouldRerun = this.forwardRerunRequested;
        this.forwardRerunRequested = false;
        if (activeSeq === this.forwardRequestSeq) {
          this.forwardBusy = false;
        }
        if (shouldRerun) {
          this.runForward(true);
        }
      }
    }, 80);
  }

  triggerForwardCompute(): void {
    this.runForward(true);
  }

  cancelForwardCompute(): void {
    this.forwardRerunRequested = false;
    this.forwardBusy = false;
    this.forwardStatusMessage = '计算已取消。';
    this.forwardRequestSeq += 1;
  }

  onAutoForwardComputeToggle(): void {
    if (this.autoForwardCompute && this.pendingForwardChanges) {
      this.runForward(true);
    }
  }

  private applyForwardResult(result: ForwardPassResult, seq: number): void {
    if (seq !== this.forwardRequestSeq) return;
    this.forwardResult = result;
    this.forwardLayerShapeMap = result.layerShapeMap;
    const hasSelected = result.layerResults.some(r => r.layerId === this.selectedLayerId);
    if (!hasSelected && result.layerResults.length) {
      this.selectedLayerId = result.layerResults[0].layerId;
    }
  }

  private buildForwardRecordSnapshot(): ForwardRecordSnapshot {
    return {
      selectedTemplateId: this.selectedTemplateId,
      selectedDataset: this.selectedDataset,
      selectedSampleId: this.selectedSampleId,
      selectedLayerId: this.selectedLayerId,
      uploadComputeProfile: this.uploadComputeProfile,
      uploadedImageUrl: this.uploadedImageUrl ? 'stored-on-spring-backend' : '',
      layers: structuredClone(this.layers),
      connections: structuredClone(this.connections),
      forwardResult: this.forwardResult ? structuredClone(this.forwardResult) : null
    };
  }

  private async applyForwardRecord(detail: ForwardRecordDetail): Promise<void> {
    const snapshot = detail.snapshot;
    this.mode = 'forward';
    this.selectedTemplateId = snapshot.selectedTemplateId;
    this.selectedDataset = snapshot.selectedDataset;
    this.selectedSampleId = snapshot.selectedSampleId;
    this.selectedLayerId = snapshot.selectedLayerId;
    this.uploadComputeProfile = snapshot.uploadComputeProfile;
    this.layers = structuredClone(snapshot.layers);
    this.connections = structuredClone(snapshot.connections);
    this.nextLayerId = Math.max(0, ...this.layers.map(layer => layer.id)) + 1;
    this.forwardResult = snapshot.forwardResult ? structuredClone(snapshot.forwardResult) : null;
    this.forwardLayerShapeMap = this.forwardResult?.layerShapeMap ?? {};
    this.forwardBackendError = '';
    this.pendingForwardChanges = false;
    this.showSamplePicker = false;

    const savedImageUrl = this.forwardRecordSvc.imageUrl(detail.imagePath);
    if (savedImageUrl) {
      await this.restoreUploadedImageFromUrl(savedImageUrl);
    } else {
      this.uploadedImageUrl = '';
      this.uploadedImageData = null;
      this.rebuildInputAsset();
    }
  }

  private async restoreUploadedImageFromUrl(imageUrl: string): Promise<void> {
    const response = await fetch(imageUrl);
    if (!response.ok) {
      throw new Error('记录图片读取失败');
    }
    const blob = await response.blob();
    const dataUrl = await new Promise<string>((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error('记录图片解析失败'));
      reader.onload = () => resolve(typeof reader.result === 'string' ? reader.result : '');
      reader.readAsDataURL(blob);
    });
    const { imageData, previewUrl } = await this.decodeAndResizeImage(dataUrl);
    this.uploadedImageUrl = previewUrl;
    this.uploadedImageData = imageData;
    this.rebuildInputAsset();
  }

  private currentInputPreviewDataUrl(): string | null {
    if (this.uploadedImageUrl) {
      return this.uploadedImageUrl;
    }
    const tensor = this.currentInputAsset?.prepared.displayTensor ?? this.currentInputAsset?.originalTensor;
    if (!tensor || tensor.shape.length !== 3) {
      return null;
    }
    const [height, width, channels] = tensor.shape as [number, number, number];
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    const image = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i += 1) {
      const src = i * channels;
      const dst = i * 4;
      const r = channels >= 3 ? tensor.values[src] ?? 0 : tensor.values[i] ?? 0;
      const g = channels >= 3 ? tensor.values[src + 1] ?? r : r;
      const b = channels >= 3 ? tensor.values[src + 2] ?? r : r;
      image.data[dst] = Math.round(Math.max(0, Math.min(1, r)) * 255);
      image.data[dst + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
      image.data[dst + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
      image.data[dst + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    return canvas.toDataURL('image/png');
  }

  // ── Training ──────────────────────────────────────────
  async loadTrainingDatasets(): Promise<void> {
    this.trainingDatasetLoading = true;
    this.trainingBackendNotice = '正在从 Spring Boot 后端加载训练数据集...';
    try {
      this.builtinTrainingDatasets = await this.trainingDatasetApi.listDatasets();
      this.trainingBackendNotice = '已连接 Spring Boot 后端。';
      await this.selectTrainingDataset(this.selectedTrainingDatasetId);
    } catch (err) {
      this.trainingBackendNotice = '后端数据集接口暂不可用，已使用前端兜底数据。';
      this.trainingDatasetError = err instanceof Error ? err.message : '加载后端数据集失败。';
      this.selectTrainingDatasetLocal(this.selectedTrainingDatasetId);
    } finally {
      this.trainingDatasetLoading = false;
    }
  }

  async selectTrainingDataset(id: string): Promise<void> {
    if (id === 'custom-upload') {
      this.useImportedTrainingDataset();
      return;
    }
    const option = this.builtinTrainingDatasets.find(d => d.id === id);
    if (!option) return;
    this.selectedTrainingDatasetId = option.id;
    this.trainingDatasetLoading = true;
    try {
      this.trainingDatasetDetail = await this.trainingDatasetApi.getDatasetDetail(option.id);
      this.trainingDatasetError = '';
      this.trainingBackendNotice = '数据集详情来自 Spring Boot 后端。';
    } catch (err) {
      this.trainingDatasetDetail = this.buildBuiltinTrainingDatasetDetail(option);
      this.trainingDatasetError = err instanceof Error ? err.message : '后端详情加载失败，已使用前端兜底数据。';
      this.trainingBackendNotice = '后端详情接口暂不可用，当前详情为前端兜底。';
    } finally {
      this.trainingDatasetLoading = false;
    }
  }

  private selectTrainingDatasetLocal(id: string): void {
    const option = this.builtinTrainingDatasets.find(d => d.id === id);
    if (!option) return;
    this.selectedTrainingDatasetId = option.id;
    this.trainingDatasetDetail = this.buildBuiltinTrainingDatasetDetail(option);
  }

  useImportedTrainingDataset(): void {
    if (!this.datasetImportDraft.detail) return;
    this.selectedTrainingDatasetId = this.datasetImportDraft.detail.id;
    this.trainingDatasetDetail = this.datasetImportDraft.detail;
    this.trainingDatasetError = '';
  }

  clearImportedTrainingDataset(): void {
    const importedId = this.datasetImportDraft.detail?.id;
    this.datasetImportDraft = {
      status: 'idle',
      message: '尚未导入自定义数据。',
      files: [],
      detectedKind: null,
      detail: null,
      csvHeaders: [],
      selectedLabelColumn: '',
      selectedClassCount: null
    };
    if (this.selectedTrainingDatasetId === 'custom-upload' || this.selectedTrainingDatasetId === importedId) {
      void this.selectTrainingDataset('mnist-1000');
    }
  }

  async onTrainingDatasetUpload(e: Event): Promise<void> {
    const input = e.target as HTMLInputElement;
    const files = Array.from(input.files ?? []);
    input.value = '';
    if (!files.length) return;

    this.datasetImportDraft = {
      status: 'idle',
      message: '正在解析本地文件...',
      files,
      detectedKind: null,
      detail: null,
      csvHeaders: [],
      selectedLabelColumn: '',
      selectedClassCount: null
    };

    try {
      const csvFiles = files.filter(file => this.isCsvFile(file));
      const zipFiles = files.filter(file => file.name.toLowerCase().endsWith('.zip') || file.type === 'application/zip' || file.type === 'application/x-zip-compressed');
      const imageFiles = files.filter(file => file.type.startsWith('image/'));
      if (zipFiles.length && files.length > zipFiles.length) {
        throw new Error('请不要混合上传 ZIP 和其他文件；ZIP 数据集一次只上传 1 个。');
      }
      if (zipFiles.length > 1) {
        throw new Error('图片 ZIP 数据集当前一次只支持上传 1 个 ZIP 文件。');
      }
      if (csvFiles.length && imageFiles.length) {
        throw new Error('请不要混合上传 CSV 和图片；一次导入只对应一种数据集类型。');
      }
      if (csvFiles.length > 1) {
        throw new Error('表格数据当前一次只支持上传 1 个 CSV 文件。');
      }
      if (!zipFiles.length && !csvFiles.length && !imageFiles.length) {
        throw new Error('仅支持 ZIP 图片数据集、CSV 文件或少量图片文件。');
      }
      if (csvFiles.length === 1) {
        const headers = await this.readCsvHeaders(csvFiles[0]);
        if (headers.length < 2) {
          throw new Error('CSV 至少需要 2 列：特征列和标签列。');
        }
        this.datasetImportDraft = {
          status: 'pending',
          message: '请选择 CSV 中哪一列作为标签列，然后再导入。',
          files,
          detectedKind: 'table',
          detail: null,
          csvHeaders: headers,
          selectedLabelColumn: '',
          selectedClassCount: null
        };
        this.trainingDatasetError = '';
        return;
      }
      await this.importTrainingDatasetFiles(files);
    } catch (err) {
      this.datasetImportDraft = {
        status: 'error',
        message: err instanceof Error ? err.message : '导入失败。',
        files,
        detectedKind: null,
        detail: null,
        csvHeaders: [],
        selectedLabelColumn: '',
        selectedClassCount: null
      };
      this.trainingDatasetError = this.datasetImportDraft.message;
    }
  }

  async confirmCsvDatasetImport(): Promise<void> {
    const files = this.datasetImportDraft.files;
    const labelColumn = this.datasetImportDraft.selectedLabelColumn;
    const classCount = Number(this.datasetImportDraft.selectedClassCount);
    if (!files.length || this.datasetImportDraft.detectedKind !== 'table') {
      this.trainingDatasetError = '请先选择一个 CSV 文件。';
      return;
    }
    if (!labelColumn) {
      this.trainingDatasetError = '请选择 CSV 标签列。';
      this.datasetImportDraft = {
        ...this.datasetImportDraft,
        status: 'error',
        message: '请选择 CSV 标签列后再导入。'
      };
      return;
    }
    if (!Number.isInteger(classCount) || classCount < 2) {
      this.trainingDatasetError = '请输入至少为 2 的类别数。';
      this.datasetImportDraft = {
        ...this.datasetImportDraft,
        status: 'error',
        message: '请输入至少为 2 的类别数后再导入。'
      };
      return;
    }
    await this.importTrainingDatasetFiles(files, labelColumn, classCount);
  }

  private async importTrainingDatasetFiles(files: File[], labelColumn?: string, classCount?: number): Promise<void> {
    try {
      this.datasetImportDraft = {
        ...this.datasetImportDraft,
        status: 'pending',
        message: '正在上传并校验数据集...'
      };
      const imported = await this.trainingDatasetApi.importDataset(files, labelColumn, classCount);
      const detail = imported.detail;
      this.upsertTrainingDatasetOption(detail);

      this.datasetImportDraft = {
        status: detail.hasLabels ? 'ready' : 'error',
        message: detail.hasLabels ? '后端已导入自定义数据，可用于训练。' : '后端已解析文件，但缺少可训练标签。',
        files,
        detectedKind: detail.kind,
        detail,
        csvHeaders: this.datasetImportDraft.csvHeaders,
        selectedLabelColumn: labelColumn ?? '',
        selectedClassCount: classCount ?? null
      };
      this.trainingDatasetDetail = detail;
      this.selectedTrainingDatasetId = detail.id;
      this.trainingDatasetError = detail.hasLabels ? '' : '当前导入数据缺少标签，训练前需要补充标签列或按类别命名图片。';
    } catch (err) {
      this.datasetImportDraft = {
        ...this.datasetImportDraft,
        status: 'error',
        message: err instanceof Error ? err.message : '导入失败。',
        detail: null
      };
      this.trainingDatasetError = this.datasetImportDraft.message;
    }
  }

  private upsertTrainingDatasetOption(detail: TrainingDatasetDetail): void {
    const option: TrainingDatasetOption = {
      id: detail.id,
      name: detail.name,
      source: detail.source,
      kind: detail.kind,
      description: detail.description,
      sampleCount: detail.sampleCount,
      classCount: detail.classCount,
      inputShape: detail.inputShape,
      recommendedSplit: detail.recommendedSplit,
      labels: detail.labels
    };
    const exists = this.builtinTrainingDatasets.some(item => item.id === option.id);
    this.builtinTrainingDatasets = exists
      ? this.builtinTrainingDatasets.map(item => item.id === option.id ? option : item)
      : [option, ...this.builtinTrainingDatasets];
  }

  async deleteUploadedTrainingDataset(detail: TrainingDatasetDetail): Promise<void> {
    if (detail.source !== 'upload') {
      this.trainingDatasetError = '内置数据集不能删除。';
      return;
    }
    const ok = window.confirm(`确定删除上传数据集“${detail.name}”吗？该操作会同时删除后端保存的文件。`);
    if (!ok) return;

    this.trainingDatasetLoading = true;
    try {
      await this.trainingDatasetApi.deleteDataset(detail.id);
      this.builtinTrainingDatasets = this.builtinTrainingDatasets.filter(item => item.id !== detail.id);
      if (this.datasetImportDraft.detail?.id === detail.id) {
        this.datasetImportDraft = {
          status: 'idle',
          message: '尚未导入自定义数据。',
          files: [],
          detectedKind: null,
          detail: null,
          csvHeaders: [],
          selectedLabelColumn: '',
          selectedClassCount: null
        };
      }
      this.trainingDatasetError = '';
      this.trainingBackendNotice = '已删除上传数据集。';
      const fallback = this.builtinTrainingDatasets.find(item => item.source === 'builtin') ?? this.builtinTrainingDatasets[0];
      if (fallback) {
        await this.selectTrainingDataset(fallback.id);
      } else {
        this.trainingDatasetDetail = null;
        this.selectedTrainingDatasetId = '';
      }
    } catch (err) {
      this.trainingDatasetError = err instanceof Error ? err.message : '删除上传数据集失败。';
    } finally {
      this.trainingDatasetLoading = false;
    }
  }

  async startTraining(): Promise<void> {
    if (!this.trainingDatasetDetail) {
      this.trainingDatasetError = '请先选择或导入一个训练数据集。';
      return;
    }
    if (!this.trainingDatasetDetail.hasLabels) {
      this.trainingDatasetError = '监督训练需要标签；请导入包含 label/class/target 列的 CSV，或用“类别_序号.jpg”命名图片。';
      return;
    }
    const splitError = this.datasetSplitError;
    if (splitError) {
      this.trainingDatasetError = splitError;
      return;
    }
    const modelError = this.trainingModelIssues.find(issue => issue.level === 'error');
    if (modelError) {
      this.trainingDatasetError = modelError.message;
      return;
    }
    this.trainingDatasetError = '';
    try {
      await this.trainingSvc.startBackend({
        datasetId: this.trainingDatasetDetail.id,
        split: {
          train: this.trainingDatasetDetail.trainRatio,
          val: this.trainingDatasetDetail.valRatio,
          test: this.trainingDatasetDetail.testRatio
        },
        layers: this.layers,
        connections: this.connections,
        config: this.trainingConfig
      });
      this.trainingBackendNotice = '训练任务由 Spring Boot 后端运行，指标通过 WebSocket 推送。';
    } catch (err) {
      this.trainingDatasetError = err instanceof Error ? err.message : '启动后端训练失败。';
    }
  }
  pauseTraining(): void  { void this.trainingSvc.pause(); }
  resumeTraining(): void { void this.trainingSvc.resume(); }
  stopTraining(): void   { void this.trainingSvc.stop(); }
  resetTraining(): void  { void this.trainingSvc.reset(); }

  async loadTrainingCheckpoints(): Promise<void> {
    if (!this.authUser) return;
    try {
      this.trainingCheckpoints = await this.trainingSvc.listCheckpoints();
      if (!this.selectedCheckpointId && this.trainingCheckpoints.length) {
        this.selectedCheckpointId = this.trainingCheckpoints[0].id;
      }
      if (this.selectedCheckpointId && !this.trainingCheckpoints.some(item => item.id === this.selectedCheckpointId)) {
        this.selectedCheckpointId = this.trainingCheckpoints[0]?.id ?? null;
      }
      this.checkpointError = '';
    } catch (err) {
      this.checkpointError = err instanceof Error ? err.message : '加载 checkpoint 失败。';
    }
  }

  async runSelectedCheckpointTest(): Promise<void> {
    if (!this.selectedCheckpointId || !this.trainingDatasetDetail) return;
    this.checkpointBusy = true;
    this.checkpointError = '';
    try {
      await this.trainingSvc.testCheckpoint(this.selectedCheckpointId, {
        datasetId: this.trainingDatasetDetail.id,
        layers: this.layers
      });
    } catch (err) {
      this.checkpointError = err instanceof Error ? err.message : 'Checkpoint 测试失败。';
    } finally {
      this.checkpointBusy = false;
    }
  }

  selectTask(id: string): void {
    this.selectedTaskId = id;
    const task = this.presetTasks.find(t => t.id === id);
    if (task?.dataset === 'MNIST') void this.selectTrainingDataset('mnist-1000');
    if (task?.dataset === 'CIFAR-10') void this.selectTrainingDataset('cifar10-5000');
  }

  runExperiments(): void {
    const task = this.presetTasks.find(t => t.id === this.selectedTaskId) ?? this.presetTasks[0];
    const base = SimEngine.evaluateTask(task, this.layers, this.trainingConfig.optimizer as any, this.trainingConfig.totalEpochs);
    this.experimentResults = [
      { name: '基准配置',       epochs: this.trainingConfig.totalEpochs, finalAccuracy: base, speedScore: 1 },
      SimEngine.runExperiment('deeper',     base, this.trainingConfig.totalEpochs),
      SimEngine.runExperiment('activation', base, this.trainingConfig.totalEpochs),
      SimEngine.runExperiment('optimizer',  base, this.trainingConfig.totalEpochs)
    ];
  }

  // ── Helpers ───────────────────────────────────────────
  layerTypeLabel(t: LayerType): string { return SimEngine.layerTypeLabel(t); }
  cellColor(v: number): string { return SimEngine.cellColor(v); }
  labelPercent(count: number): number { return Math.max(5, (count / this.trainingDatasetMaxLabelCount) * 100); }
  pointSvgX(point: PointPreviewItem): number { return 12 + ((point.x + 1) / 2) * 176; }
  pointSvgY(point: PointPreviewItem): number { return 108 - ((point.y + 1) / 2) * 96; }
  formatDuration(seconds: number): string {
    const safe = Math.max(0, Math.round(seconds));
    const m = Math.floor(safe / 60);
    const s = safe % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }
  datasetSplitPercent(kind: 'train' | 'val' | 'test'): number {
    const ds = this.trainingDatasetDetail;
    if (!ds) return 0;
    const ratio = kind === 'train' ? ds.trainRatio : kind === 'val' ? ds.valRatio : ds.testRatio;
    return Math.round(ratio * 1000) / 10;
  }

  applyDatasetSplitPreset(train: number, val: number, test: number): void {
    const ds = this.trainingDatasetDetail;
    if (!ds) return;
    ds.trainRatio = train / 100;
    ds.valRatio = val / 100;
    ds.testRatio = test / 100;
  }

  onDatasetSplitInput(kind: 'train' | 'val' | 'test', rawValue: string | number): void {
    const ds = this.trainingDatasetDetail;
    if (!ds) return;
    const value = typeof rawValue === 'number' ? rawValue : Number(rawValue);
    const ratio = Number.isFinite(value) ? value / 100 : 0;
    if (kind === 'train') ds.trainRatio = ratio;
    if (kind === 'val') ds.valRatio = ratio;
    if (kind === 'test') ds.testRatio = ratio;
  }

  private analyzeTrainingNetwork(): { issues: LayerValidationIssue[]; shapeMap: Record<number, string> } {
    const issues: LayerValidationIssue[] = [];
    const shapeMap: Record<number, string> = {};
    const enabledLayers = this.layers.filter(layer => layer.enabled !== false);
    let currentShape: TensorShape = this.trainingInputShape();

    const addIssue = (
      layer: NetworkLayer,
      severity: 'error' | 'warning',
      message: string,
      field?: string
    ): void => {
      issues.push({ layerId: layer.id, layerName: layer.name, severity, message, field });
    };
    const intValue = (value: unknown): number => typeof value === 'number' ? value : Number(value);
    const isIntAtLeast = (value: unknown, min: number): boolean => {
      const n = intValue(value);
      return Number.isFinite(n) && Number.isInteger(n) && n >= min;
    };
    const isNumberInRange = (value: unknown, min: number, max: number, inclusiveMax = true): boolean => {
      const n = intValue(value);
      return Number.isFinite(n) && n >= min && (inclusiveMax ? n <= max : n < max);
    };

    const inputIndex = enabledLayers.findIndex(layer => layer.type === 'input');
    const outputIndex = enabledLayers.findIndex(layer => layer.type === 'output');
    if (inputIndex > 0) {
      addIssue(enabledLayers[inputIndex], 'error', '输入层必须位于网络第一层。');
    }
    if (outputIndex >= 0 && outputIndex !== enabledLayers.length - 1) {
      addIssue(enabledLayers[outputIndex], 'error', '输出层必须位于网络最后一层。');
    }

    for (const layer of enabledLayers) {
      if (!layer.name.trim()) {
        addIssue(layer, 'warning', '层名称为空，建议补充一个可读名称。');
      }

      if (layer.type === 'input') {
        const p = layer.params;
        if (p.inputKind === 'table') {
          if (!isIntAtLeast(p.featureCount, 1)) addIssue(layer, 'error', 'CSV 特征数必须是大于 0 的整数。', 'featureCount');
          const expected = this.trainingInputShape();
          if (this.trainingDatasetDetail?.kind !== 'image' && expected.length === 1 && isIntAtLeast(p.featureCount, 1) && intValue(p.featureCount) !== expected[0]) {
            addIssue(layer, 'warning', `当前输入特征数为 ${p.featureCount}，数据集预估为 ${expected[0]} 个特征。`, 'featureCount');
          }
          currentShape = isIntAtLeast(p.featureCount, 1) ? [Math.floor(intValue(p.featureCount))] : [];
        } else {
          if (!isIntAtLeast(p.height, 1)) addIssue(layer, 'error', '输入高度必须是大于 0 的整数。', 'height');
          if (!isIntAtLeast(p.width, 1)) addIssue(layer, 'error', '输入宽度必须是大于 0 的整数。', 'width');
          if (!isIntAtLeast(p.channels, 1)) addIssue(layer, 'error', '输入通道数必须是大于 0 的整数。', 'channels');
          currentShape = isIntAtLeast(p.height, 1) && isIntAtLeast(p.width, 1) && isIntAtLeast(p.channels, 1)
            ? [Math.floor(intValue(p.height)), Math.floor(intValue(p.width)), Math.floor(intValue(p.channels))]
            : [];
        }
      } else if (layer.type === 'conv2d') {
        if (currentShape.length !== 3) {
          addIssue(layer, 'error', `Conv2D 需要 3D 图像或特征图输入，当前输入为 ${SimEngine.formatShapeLabel(currentShape)}。`);
          currentShape = [];
        } else {
          if (!isIntAtLeast(layer.params.outChannels, 1)) addIssue(layer, 'error', '输出通道必须是大于 0 的整数。', 'outChannels');
          if (!isIntAtLeast(layer.params.kernelSize, 1)) addIssue(layer, 'error', '卷积核大小必须是大于 0 的整数。', 'kernelSize');
          if (!isIntAtLeast(layer.params.stride, 1)) addIssue(layer, 'error', '步长必须是大于 0 的整数。', 'stride');
          if (!isIntAtLeast(layer.params.padding, 0)) addIssue(layer, 'error', '填充必须是非负整数。', 'padding');
          if (!isIntAtLeast(layer.params.dilation, 1)) addIssue(layer, 'error', '膨胀率必须是大于 0 的整数。', 'dilation');
          const [h, w] = currentShape;
          const k = Math.floor(intValue(layer.params.kernelSize));
          const p = Math.floor(intValue(layer.params.padding));
          const d = Math.floor(intValue(layer.params.dilation));
          const s = Math.floor(intValue(layer.params.stride));
          const effectiveKernel = d * (k - 1) + 1;
          const outH = Math.floor((h + 2 * p - effectiveKernel) / s) + 1;
          const outW = Math.floor((w + 2 * p - effectiveKernel) / s) + 1;
          if (Number.isFinite(outH) && Number.isFinite(outW) && (outH <= 0 || outW <= 0)) {
            addIssue(layer, 'error', `卷积后空间尺寸为 ${outH}x${outW}，请减小核大小/膨胀率或增大填充。`, 'kernelSize');
            currentShape = [];
          } else {
            currentShape = SimEngine.inferLayerOutputShape(layer, [currentShape]);
          }
        }
      } else if (layer.type === 'pool2d') {
        if (currentShape.length !== 3) {
          addIssue(layer, 'error', `池化层需要 3D 图像或特征图输入，当前输入为 ${SimEngine.formatShapeLabel(currentShape)}。`);
          currentShape = [];
        } else {
          if (!isIntAtLeast(layer.params.kernelSize, 1)) addIssue(layer, 'error', '池化核大小必须是大于 0 的整数。', 'kernelSize');
          if (!isIntAtLeast(layer.params.stride, 1)) addIssue(layer, 'error', '池化步长必须是大于 0 的整数。', 'stride');
          if (!isIntAtLeast(layer.params.padding, 0)) addIssue(layer, 'error', '池化填充必须是非负整数。', 'padding');
          if (isIntAtLeast(layer.params.kernelSize, 1) && isIntAtLeast(layer.params.padding, 0)
            && intValue(layer.params.padding) > Math.floor(intValue(layer.params.kernelSize) / 2)) {
            addIssue(layer, 'error', 'PyTorch 池化填充不能大于核大小的一半。', 'padding');
          }
          const nextShape = SimEngine.inferLayerOutputShape(layer, [currentShape]);
          if (!nextShape.length) {
            addIssue(layer, 'error', '池化后空间尺寸小于 1，请减小核大小或调整步长/填充。', 'kernelSize');
          }
          currentShape = nextShape;
        }
      } else if (layer.type === 'residual') {
        if (currentShape.length !== 3) {
          addIssue(layer, 'error', `残差块需要 3D 图像或特征图输入，当前输入为 ${SimEngine.formatShapeLabel(currentShape)}。`);
          currentShape = [];
        } else {
          if (!isIntAtLeast(layer.params.outChannels, 1)) addIssue(layer, 'error', '输出通道必须是大于 0 的整数。', 'outChannels');
          if (!isIntAtLeast(layer.params.kernelSize, 1)) addIssue(layer, 'error', '卷积核大小必须是大于 0 的整数。', 'kernelSize');
          if (!isIntAtLeast(layer.params.stride, 1)) addIssue(layer, 'error', '步长必须是大于 0 的整数。', 'stride');
          if (!isIntAtLeast(layer.params.padding, 0)) addIssue(layer, 'error', '填充必须是非负整数。', 'padding');

          const nextShape = SimEngine.inferLayerOutputShape(layer, [currentShape]);
          if (!nextShape.length) {
            addIssue(layer, 'error', '残差主分支输出尺寸小于 1，请减小核大小/步长或调整填充。', 'kernelSize');
            currentShape = [];
          } else {
            const projectionShape = this.residualProjectionShape(currentShape, Math.max(1, Math.floor(intValue(layer.params.stride))), Math.max(1, Math.floor(intValue(layer.params.outChannels))));
            const skipShape = layer.params.useProjection ? projectionShape : currentShape;
            if (!this.sameShape(skipShape, nextShape)) {
              const shortcutLabel = layer.params.useProjection ? '1x1 投影分支' : 'shortcut 分支';
              addIssue(
                layer,
                'error',
                `${shortcutLabel}为 ${SimEngine.formatShapeLabel(skipShape)}，主分支为 ${SimEngine.formatShapeLabel(nextShape)}，两路相加前维度必须一致。`,
                'useProjection'
              );
            }
            currentShape = nextShape;
          }
        }
      } else if (layer.type === 'flatten') {
        if (!currentShape.length) addIssue(layer, 'error', '上一层输出形状无效，无法展开。');
        currentShape = SimEngine.inferLayerOutputShape(layer, [currentShape]);
      } else if (layer.type === 'dense') {
        if (!isIntAtLeast(layer.params.units, 1)) addIssue(layer, 'error', '神经元数必须是大于 0 的整数。', 'units');
        if (!currentShape.length) addIssue(layer, 'error', '上一层输出形状无效，无法连接全连接层。');
        currentShape = isIntAtLeast(layer.params.units, 1) ? [Math.floor(intValue(layer.params.units))] : [];
      } else if (layer.type === 'activation') {
        if (!currentShape.length) addIssue(layer, 'error', '上一层输出形状无效，无法应用激活函数。');
      } else if (layer.type === 'dropout') {
        if (!isNumberInRange(layer.params.rate, 0, 1, false)) {
          addIssue(layer, 'error', 'Dropout 比率必须在 [0, 1) 范围内。', 'rate');
        }
        if (!currentShape.length) addIssue(layer, 'error', '上一层输出形状无效，无法应用 Dropout。');
      } else if (layer.type === 'output') {
        if (!isIntAtLeast(layer.params.units, 1)) addIssue(layer, 'error', '输出类别数必须是大于 0 的整数。', 'units');
        if (!currentShape.length) addIssue(layer, 'error', '上一层输出形状无效，无法连接输出层。');
        currentShape = isIntAtLeast(layer.params.units, 1) ? [Math.floor(intValue(layer.params.units))] : [];
      }

      shapeMap[layer.id] = SimEngine.formatShapeLabel(currentShape);
    }

    return { issues, shapeMap };
  }

  private residualProjectionShape(inputShape: TensorShape, stride: number, outChannels: number): TensorShape {
    if (inputShape.length !== 3) return [];
    const [h, w] = inputShape;
    const safeStride = Math.max(1, stride);
    const outH = Math.floor((h - 1) / safeStride) + 1;
    const outW = Math.floor((w - 1) / safeStride) + 1;
    return outH > 0 && outW > 0 ? [outH, outW, Math.max(1, outChannels)] : [];
  }

  private sameShape(a: TensorShape, b: TensorShape): boolean {
    return a.length === b.length && a.every((value, index) => value === b[index]);
  }

  private trainingInputShape(): TensorShape {
    const ds = this.trainingDatasetDetail;
    if (ds?.kind === 'image') {
      const p = this.inputLayer?.params;
      if (p && Number.isFinite(+p.height) && Number.isFinite(+p.width) && Number.isFinite(+p.channels)) {
        return [Math.max(1, Math.floor(+p.height)), Math.max(1, Math.floor(+p.width)), Math.max(1, Math.floor(+p.channels))];
      }
      const parsed = ds.inputShape.match(/(\d+)\s*x\s*(\d+)\s*x\s*(\d+)/i);
      if (parsed) return [Number(parsed[1]), Number(parsed[2]), Number(parsed[3])];
      return [32, 32, 3];
    }
    if (ds?.kind === 'points') return [2];
    const featureMatch = ds?.inputShape.match(/(\d+)/);
    return [featureMatch ? Math.max(1, Number(featureMatch[1])) : 1];
  }

  private syncTemplateWithTrainingDataset(): void {
    const ds = this.trainingDatasetDetail;
    if (!ds) return;
    const input = this.layers.find((layer): layer is InputLayer => layer.type === 'input');
    const output = this.layers.find(layer => layer.type === 'output');
    if (input) {
      if (ds.kind === 'image') {
        input.params.inputKind = 'image';
        const parsed = ds.inputShape.match(/(\d+)\s*x\s*(\d+)\s*x\s*(\d+)/i);
        if (parsed) {
          input.params.height = Number(parsed[1]);
          input.params.width = Number(parsed[2]);
          input.params.channels = Number(parsed[3]);
        }
      } else {
        const shape = this.trainingInputShape();
        input.params.inputKind = 'table';
        input.params.featureCount = shape.length === 1 ? shape[0] : Math.max(1, input.params.featureCount ?? 1);
      }
    }
    if (output?.type === 'output' && ds.classCount > 0) {
      output.params.units = ds.classCount;
    }
  }

  private normBars(vals: number[]): number[] {
    if (!vals.length) return [];
    let mn = Number.POSITIVE_INFINITY;
    let mx = Number.NEGATIVE_INFINITY;
    for (const v of vals) {
      if (v < mn) mn = v;
      if (v > mx) mx = v;
    }
    const sp = Math.max(1e-6, mx - mn);
    if (sp <= 1e-6) return vals.map(() => 0.55);
    return vals.map(v => (v - mn) / sp);
  }

  onUploadComputeProfileChange(): void {
    const imageData = this.uploadedImageData ?? this.localImageData;
    if (!imageData) return;
    this.applyUploadComputeProfile(imageData.width, imageData.height);
    this.rebuildInputAsset();
    this.runForward();
  }

  private rebuildTopology(): void {
    this.layers = this.layers.map((l, i) => ({ ...l, inputs: i === 0 ? [] : [this.layers[i - 1].id] }));
    this.connections = SimEngine.rebuildLinearConnections(this.layers);
    this.syncKernelShape();
  }

  private syncKernelShape(): void {
    const l = this.selectedLayer;
    if (!l || l.type !== 'conv2d') return;
    this.ensureConvKernelBank(l);
    const sz = Math.max(1, Math.floor(l.params.kernelSize));
    l.params.kernelSize = sz;
    const outChannels = Math.max(1, Math.floor(l.params.outChannels));
    const inChannels = this.selectedConvInChannels;
    const src = l.params.kernels ?? [];
    l.params.kernels = Array.from({ length: outChannels }, (_, oc) => {
      const srcKernel = src[oc]?.weights ?? [];
      const weights = Array.from({ length: inChannels }, (_, ic) => {
        const srcMatrix = srcKernel[ic] ?? l.params.kernelMatrix ?? [];
        return Array.from({ length: sz }, (_, r) =>
          Array.from({ length: sz }, (_, c) => srcMatrix[r]?.[c] ?? 0)
        );
      });
      return { weights } as ConvKernelSpec;
    });
    l.params.kernelMatrix = l.params.kernels?.[0]?.weights?.[0]?.map(row => [...row]) ?? [];
    this.syncConvKernelSelectors();
  }

  private rebuildInputAsset(): void {
    const pre = this.inputLayer?.params.preprocessing;
    if (!pre) { this.currentInputAsset = null; return; }
    if (this.uploadedImageData) {
      this.currentInputAsset = SimEngine.createForwardInputAssetFromImageData({
        id: 'upload', name: '上传图片', source: 'upload',
        imageData: this.uploadedImageData, preprocess: pre, previewUrl: this.uploadedImageUrl
      });
    } else if (this.localImageData) {
      const sample = this.localImageSamples.find(item => item.id === this.selectedLocalImageId);
      this.currentInputAsset = SimEngine.createForwardInputAssetFromImageData({
        id: sample?.id ?? 'local-sample',
        name: sample?.name ?? '本地示例',
        source: 'dataset',
        imageData: this.localImageData,
        preprocess: pre,
        previewUrl: this.localImagePreviewUrl,
        label: sample?.label
      });
    } else {
      const s = this.selectedSample;
      if (!s) { this.currentInputAsset = null; return; }
      this.currentInputAsset = SimEngine.createForwardInputAssetFromSample(s, pre);
    }
    this.syncInputShape();
  }

  private async loadLocalImageSamples(): Promise<void> {
    try {
      const response = await fetch('a-mode-samples/manifest.json');
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      this.localImageSamples = await response.json() as LocalImageSample[];
      if (!this.localImageSamples.some(sample => sample.category === this.selectedDataset)) {
        this.selectedDataset = this.localImageSamples[0]?.category ?? this.selectedDataset;
      }
      const first = this.activeLocalImageSamples[0] ?? this.localImageSamples[0];
      if (first) {
        this.selectedDataset = first.category;
        await this.chooseLocalImageSample(first);
      }
    } catch {
      this.localImageError = '本地示例图片清单加载失败';
    }
  }

  private clearLocalImageSelection(): void {
    this.selectedLocalImageId = '';
    this.localImageData = null;
    this.localImagePreviewUrl = '';
    this.localImageError = '';
  }

  private syncInputShape(): void {
    const il = this.inputLayer, t = this.currentInputAsset?.prepared.tensor;
    if (!il || !t || t.shape.length !== 3) return;
    il.params.height = t.shape[0]; il.params.width = t.shape[1]; il.params.channels = t.shape[2];
    il.params.colorMode = t.shape[2] === 1 ? 'grayscale' : 'rgb';
  }

  private extractChannel(values: number[], h: number, w: number, c: number, channel: number): number[] {
    const out = new Array(h * w);
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const idx = ((y * w) + x) * c + channel;
        out[y * w + x] = values[idx] ?? 0;
      }
    }
    return out;
  }

  private normalizeChannel(values: number[]): number[] {
    if (!values.length) return [];
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const value of values) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
    const span = Math.max(1e-6, max - min);
    return values.map(v => (v - min) / span);
  }

  private sampleTensorForPreview(tensor: ForwardTensor, maxSide: number): ForwardTensor {
    if (tensor.shape.length !== 3) return tensor;
    const [h, w, c] = tensor.shape;
    if (Math.max(h, w) <= maxSide) return tensor;
    const scale = maxSide / Math.max(h, w);
    const outH = Math.max(1, Math.round(h * scale));
    const outW = Math.max(1, Math.round(w * scale));
    const out = new Array(outH * outW * c);
    for (let y = 0; y < outH; y += 1) {
      const srcY = Math.min(h - 1, Math.floor((y / outH) * h));
      for (let x = 0; x < outW; x += 1) {
        const srcX = Math.min(w - 1, Math.floor((x / outW) * w));
        for (let ch = 0; ch < c; ch += 1) {
          const src = ((srcY * w) + srcX) * c + ch;
          const dst = ((y * outW) + x) * c + ch;
          out[dst] = tensor.values[src] ?? 0;
        }
      }
    }
    return { ...tensor, shape: [outH, outW, c], values: out };
  }

  private previewTensorForGrid(tensor: ForwardTensor): { mode: 'rgb' | 'gray'; width: number; height: number; colors?: string[]; values?: number[] } {
    const cached = this.tensorPreviewCache.get(tensor);
    if (cached) return cached;
    const previewTensor = this.sampleTensorForPreview(tensor, MAX_PREVIEW_GRID_SIDE);
    const [h, w, c] = previewTensor.shape as [number, number, number];
    const srcValues = previewTensor.values;
    const built = c === 3
      ? {
          mode: 'rgb' as const,
          colors: Array.from({ length: h * w }, (_, i) => {
            const base = i * 3;
            return `rgb(${Math.round((srcValues[base] ?? 0) * 255)},${Math.round((srcValues[base + 1] ?? 0) * 255)},${Math.round((srcValues[base + 2] ?? 0) * 255)})`;
          }),
          width: w,
          height: h
        }
      : {
          mode: 'gray' as const,
          values: Array.from({ length: h * w }, (_, i) => srcValues[i] ?? 0),
          width: w,
          height: h
        };
    this.tensorPreviewCache.set(tensor, built);
    return built;
  }

  private tensorToImageDataUrl(tensor: ForwardTensor | null, normalize = false): string {
    if (!tensor || tensor.shape.length !== 3) return '';
    const cached = this.tensorImagePreviewCache.get(tensor);
    if (cached && !normalize) return cached;
    const [height, width, channels] = tensor.shape as [number, number, number];
    const values = tensor.values;
    const grayValues = channels === 1 || normalize
      ? this.normalizeChannel(this.extractChannel(values, height, width, channels, 0))
      : [];
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';
    const image = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i += 1) {
      const src = i * channels;
      const dst = i * 4;
      const r = channels >= 3 && !normalize ? values[src] ?? 0 : grayValues[i] ?? values[i] ?? 0;
      const g = channels >= 3 && !normalize ? values[src + 1] ?? r : r;
      const b = channels >= 3 && !normalize ? values[src + 2] ?? r : r;
      image.data[dst] = Math.round(Math.max(0, Math.min(1, r)) * 255);
      image.data[dst + 1] = Math.round(Math.max(0, Math.min(1, g)) * 255);
      image.data[dst + 2] = Math.round(Math.max(0, Math.min(1, b)) * 255);
      image.data[dst + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    const url = canvas.toDataURL('image/png');
    if (!normalize) this.tensorImagePreviewCache.set(tensor, url);
    return url;
  }

  private grayValuesToImageDataUrl(values: number[], width: number, height: number): string {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return '';
    const image = ctx.createImageData(width, height);
    for (let i = 0; i < width * height; i += 1) {
      const value = Math.round(Math.max(0, Math.min(1, values[i] ?? 0)) * 255);
      const dst = i * 4;
      image.data[dst] = value;
      image.data[dst + 1] = value;
      image.data[dst + 2] = value;
      image.data[dst + 3] = 255;
    }
    ctx.putImageData(image, 0, 0);
    return canvas.toDataURL('image/png');
  }

  private buildChannelPreviews(tensor: ForwardTensor, limit?: number): ChannelPreviewItem[] {
    if (tensor.shape.length !== 3) return [];
    const cachedAll = this.tensorChannelPreviewCache.get(tensor);
    if (cachedAll) return typeof limit === 'number' ? cachedAll.slice(0, limit) : cachedAll;

    const [h, w, c] = tensor.shape as [number, number, number];
    const all = Array.from({ length: c }, (_, ch) => ({
      channel: ch,
      width: w,
      height: h,
      values: this.normalizeChannel(this.extractChannel(tensor.values, h, w, c, ch))
    }));
    this.tensorChannelPreviewCache.set(tensor, all);
    return typeof limit === 'number' ? all.slice(0, limit) : all;
  }

  private applyUploadComputeProfile(width: number, height: number): void {
    const input = this.inputLayer;
    if (!input) return;
    const pre = input.params.preprocessing;
    pre.colorMode = 'original';

    if (this.uploadComputeProfile === 'original') {
      pre.resizeMode = 'none';
      pre.targetWidth = width;
      pre.targetHeight = height;
      return;
    }

    const maxSide = this.uploadComputeProfile === 'fast'
      ? 112
      : this.uploadComputeProfile === 'balanced'
        ? 160
        : 256;
    const scale = Math.min(1, maxSide / Math.max(width, height, 1));
    pre.resizeMode = 'fit';
    pre.targetWidth = Math.max(1, Math.round(width * scale));
    pre.targetHeight = Math.max(1, Math.round(height * scale));
  }

  private syncConvKernelSelectors(): void {
    const layer = this.selectedConvLayer;
    if (!layer) return;
    const outMax = Math.max(1, layer.params.outChannels);
    const inMax = Math.max(1, this.selectedConvInChannels);
    this.selectedKernelOutChannel = Math.min(this.selectedKernelOutChannel, outMax - 1);
    this.selectedKernelInChannel = Math.min(this.selectedKernelInChannel, inMax - 1);
    this.selectedKernelOutChannel = Math.max(0, this.selectedKernelOutChannel);
    this.selectedKernelInChannel = Math.max(0, this.selectedKernelInChannel);
  }

  private ensureConvKernelBank(layer: Extract<NetworkLayer, { type: 'conv2d' }>): void {
    const k = Math.max(1, layer.params.kernelSize);
    const outChannels = Math.max(1, layer.params.outChannels);
    const inChannels = Math.max(1, this.selectedConvInChannels);
    const current = layer.params.kernels ?? [];
    const base = layer.params.kernelMatrix ?? Array.from({ length: k }, () => Array.from({ length: k }, () => 0));
    layer.params.kernels = Array.from({ length: outChannels }, (_, oc) => {
      const srcWeights = current[oc]?.weights ?? [];
      const weights = Array.from({ length: inChannels }, (_, ic) => {
        const src = srcWeights[ic] ?? srcWeights[0] ?? base;
        return Array.from({ length: k }, (_, y) => Array.from({ length: k }, (_, x) => src[y]?.[x] ?? 0));
      });
      return { ...current[oc], weights };
    });
    layer.params.kernelMatrix = layer.params.kernels[0].weights[0].map(row => [...row]);
  }

  /**
   * 解码图片并自动缩放到 MAX_IMAGE_SIDE，防止大图卡死主线程。
   * 带超时保护，避免损坏图片永久挂起。
   */
  private decodeAndResizeImage(url: string): Promise<{ imageData: ImageData; previewUrl: string }> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      const timer = setTimeout(() => reject(new Error('图片加载超时')), IMAGE_DECODE_TIMEOUT);

      img.onerror = () => { clearTimeout(timer); reject(new Error('图片格式无效或已损坏')); };
      img.onload = () => {
        clearTimeout(timer);
        try {
          // 计算缩放比例，限制最大边长
          const scale = Math.min(1, MAX_IMAGE_SIDE / Math.max(img.naturalWidth, img.naturalHeight, 1));
          const w = Math.max(1, Math.round(img.naturalWidth * scale));
          const h = Math.max(1, Math.round(img.naturalHeight * scale));

          const canvas = document.createElement('canvas');
          canvas.width = w; canvas.height = h;
          const ctx = canvas.getContext('2d');
          if (!ctx) { reject(new Error('无法获取 Canvas 上下文')); return; }
          ctx.drawImage(img, 0, 0, w, h);
          const imageData = ctx.getImageData(0, 0, w, h);
          const previewUrl = canvas.toDataURL('image/jpeg', 0.85);
          resolve({ imageData, previewUrl });
        } catch (err) {
          reject(err);
        }
      };
      img.src = url;
    });
  }

  private buildBuiltinTrainingDatasetDetail(option: TrainingDatasetOption): TrainingDatasetDetail {
    const base = {
      ...option,
      hasLabels: true,
      trainRatio: option.id === 'iris' ? 0.8 : 0.7,
      valRatio: option.id === 'iris' ? 0 : 0.15,
      testRatio: option.id === 'iris' ? 0.2 : 0.15,
      labelDistribution: this.evenDistribution(option.labels, option.sampleCount),
      warnings: []
    };

    if (option.id === 'mnist-1000') {
      return {
        ...base,
        imagePreview: option.labels.slice(0, 8).map((label, i) => ({
          name: `mnist_${label}_${i}.png`,
          label,
          url: this.svgThumb(label, '#111827', '#f8fafc')
        }))
      };
    }

    if (option.id === 'cifar10-500' || option.id === 'cifar10-5000') {
      return {
        ...base,
        imagePreview: option.labels.slice(0, 8).map((label, i) => ({
          name: `${label}_${i}.png`,
          label,
          url: this.svgThumb(label.slice(0, 2).toUpperCase(), DATASET_COLORS[i % DATASET_COLORS.length], '#e0f2fe')
        }))
      };
    }

    if (option.id === 'iris') {
      return {
        ...base,
        tablePreview: {
          headers: ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'label'],
          rows: [
            ['5.1', '3.5', '1.4', '0.2', 'setosa'],
            ['6.4', '3.2', '4.5', '1.5', 'versicolor'],
            ['6.3', '3.3', '6.0', '2.5', 'virginica'],
            ['5.8', '2.7', '4.1', '1.0', 'versicolor']
          ]
        }
      };
    }

    return {
      ...base,
      pointPreview: this.makePointPreview()
    };
  }

  private async buildUploadedCsvDataset(file: File): Promise<TrainingDatasetDetail> {
    const text = await file.text();
    const lines = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
    if (lines.length < 2) {
      throw new Error('CSV 至少需要表头和一行数据。');
    }
    const headers = this.parseCsvLine(lines[0]);
    const rows = lines.slice(1).map(line => this.parseCsvLine(line)).filter(row => row.length > 0);
    const labelIndex = this.detectLabelColumn(headers);
    const hasLabels = labelIndex >= 0 && rows.some(row => !!row[labelIndex]?.trim());
    const labelCounts = new Map<string, number>();
    let missingValues = 0;

    for (const row of rows) {
      for (let i = 0; i < headers.length; i += 1) {
        if ((row[i] ?? '').trim() === '') missingValues += 1;
      }
      if (hasLabels) {
        const label = (row[labelIndex] ?? '').trim() || '未标注';
        labelCounts.set(label, (labelCounts.get(label) ?? 0) + 1);
      }
    }

    const labels = hasLabels ? [...labelCounts.keys()] : [];
    const warnings: string[] = [];
    if (!hasLabels) warnings.push('未检测到 label/class/target 标签列，监督训练会被阻止。');
    if (missingValues > 0) warnings.push(`发现 ${missingValues} 个缺失值，后端训练前需要清洗或填补。`);
    warnings.push(...this.imbalanceWarnings(labelCounts));

    return {
      id: `upload-${Date.now()}`,
      name: file.name,
      source: 'upload',
      kind: 'table',
      description: '本地 CSV 导入数据，当前仅在前端完成结构解析。',
      sampleCount: rows.length,
      classCount: labels.length,
      inputShape: `${Math.max(0, headers.length - (hasLabels ? 1 : 0))} columns`,
      recommendedSplit: '70% / 15% / 15%',
      labels,
      hasLabels,
      trainRatio: 0.7,
      valRatio: 0.15,
      testRatio: 0.15,
      labelDistribution: this.mapToDistribution(labelCounts),
      tablePreview: { headers, rows: rows.slice(0, 6) },
      warnings
    };
  }

  private async buildUploadedImageDataset(files: File[]): Promise<TrainingDatasetDetail> {
    const previewFiles = files.slice(0, 12);
    const previews = await Promise.all(previewFiles.map(file => this.readImagePreview(file)));
    const sizeSet = new Set(previews.map(item => `${item.width}x${item.height}`));
    const labelCounts = new Map<string, number>();
    for (const file of files) {
      const label = this.labelFromImageName(file.name);
      labelCounts.set(label, (labelCounts.get(label) ?? 0) + 1);
    }
    const labels = [...labelCounts.keys()].filter(label => label !== '未标注');
    const hasLabels = labels.length > 0 && !labelCounts.has('未标注');
    const warnings: string[] = [];
    if (!hasLabels) warnings.push('图片文件名未形成完整标签，建议使用 “类别_序号.jpg” 命名。');
    if (sizeSet.size > 1) warnings.push('检测到图片尺寸不一致，后端导入时需要统一 resize。');
    warnings.push(...this.imbalanceWarnings(labelCounts));

    return {
      id: `upload-${Date.now()}`,
      name: `图片导入 ${files.length} 张`,
      source: 'upload',
      kind: 'image',
      description: '本地图片导入数据，文件名用于前端推断类别标签。',
      sampleCount: files.length,
      classCount: labels.length,
      inputShape: sizeSet.size === 1 ? `${[...sizeSet][0]} x 3` : 'mixed image sizes',
      recommendedSplit: '70% / 15% / 15%',
      labels,
      hasLabels,
      trainRatio: 0.7,
      valRatio: 0.15,
      testRatio: 0.15,
      labelDistribution: this.mapToDistribution(labelCounts),
      imagePreview: previews.map(item => ({ name: item.name, label: item.label, url: item.url })),
      warnings
    };
  }

  private isCsvFile(file: File): boolean {
    return file.type === 'text/csv'
      || file.type === 'application/vnd.ms-excel'
      || file.name.toLowerCase().endsWith('.csv');
  }

  private async readCsvHeaders(file: File): Promise<string[]> {
    const text = await file.text();
    const firstLine = text.split(/\r?\n/).find(line => line.trim().length > 0);
    if (!firstLine) return [];
    return this.parseCsvLine(firstLine).filter(header => header.trim().length > 0);
  }

  private parseCsvLine(line: string): string[] {
    const cells: string[] = [];
    let current = '';
    let quoted = false;
    for (let i = 0; i < line.length; i += 1) {
      const ch = line[i];
      if (ch === '"') {
        if (quoted && line[i + 1] === '"') {
          current += '"';
          i += 1;
        } else {
          quoted = !quoted;
        }
      } else if (ch === ',' && !quoted) {
        cells.push(current.trim());
        current = '';
      } else {
        current += ch;
      }
    }
    cells.push(current.trim());
    return cells;
  }

  private detectLabelColumn(headers: string[]): number {
    const names = headers.map(h => h.trim().toLowerCase());
    const candidates = ['label', 'labels', 'class', 'category', 'target', 'y', '标签', '类别'];
    const exact = names.findIndex(name => candidates.includes(name));
    if (exact >= 0) return exact;
    return names.findIndex(name => candidates.some(candidate => name.includes(candidate)));
  }

  private readImagePreview(file: File): Promise<ImagePreviewItem & { width: number; height: number }> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onerror = () => reject(new Error(`读取图片失败：${file.name}`));
      reader.onload = () => {
        const url = typeof reader.result === 'string' ? reader.result : '';
        if (!url) {
          reject(new Error(`图片内容为空：${file.name}`));
          return;
        }
        const img = new Image();
        img.onerror = () => reject(new Error(`无法解析图片尺寸：${file.name}`));
        img.onload = () => resolve({
          name: file.name,
          label: this.labelFromImageName(file.name),
          url,
          width: img.naturalWidth,
          height: img.naturalHeight
        });
        img.src = url;
      };
      reader.readAsDataURL(file);
    });
  }

  private labelFromImageName(name: string): string {
    const base = name.replace(/\.[^.]+$/, '');
    const match = base.match(/^([A-Za-z0-9\u4e00-\u9fa5]+)[_-]/);
    return match?.[1] ?? '未标注';
  }

  private evenDistribution(labels: string[], sampleCount: number): LabelDistributionItem[] {
    const base = Math.floor(sampleCount / Math.max(1, labels.length));
    const extra = sampleCount - base * labels.length;
    return labels.map((label, i) => ({
      label,
      count: base + (i < extra ? 1 : 0),
      color: DATASET_COLORS[i % DATASET_COLORS.length]
    }));
  }

  private mapToDistribution(counts: Map<string, number>): LabelDistributionItem[] {
    return [...counts.entries()].map(([label, count], i) => ({
      label,
      count,
      color: DATASET_COLORS[i % DATASET_COLORS.length]
    }));
  }

  private imbalanceWarnings(counts: Map<string, number>): string[] {
    const values = [...counts.values()].filter(count => count > 0);
    if (values.length < 2) return [];
    const min = Math.min(...values);
    const max = Math.max(...values);
    return max >= min * 3 ? ['类别数量差异较大，训练结果可能偏向多数类。'] : [];
  }

  private makePointPreview(): PointPreviewItem[] {
    return Array.from({ length: 36 }, (_, i) => {
      const label = i % 2 === 0 ? 'class A' : 'class B';
      const ring = Math.floor(i / 2);
      const angle = (ring * 0.72) + (i % 2) * 0.45;
      const radius = i % 2 === 0 ? 0.32 + (ring % 5) * 0.045 : 0.66 + (ring % 4) * 0.035;
      return {
        x: Math.cos(angle) * radius,
        y: Math.sin(angle) * radius,
        label,
        color: i % 2 === 0 ? DATASET_COLORS[0] : DATASET_COLORS[3]
      };
    });
  }

  private svgThumb(text: string, fg: string, bg: string): string {
    const svg = `<svg xmlns="http://www.w3.org/2000/svg" width="80" height="80" viewBox="0 0 80 80"><rect width="80" height="80" rx="10" fill="${bg}"/><circle cx="40" cy="40" r="27" fill="${fg}" opacity=".13"/><text x="40" y="48" text-anchor="middle" font-family="Arial" font-size="24" font-weight="700" fill="${fg}">${text}</text></svg>`;
    return `data:image/svg+xml;charset=UTF-8,${encodeURIComponent(svg)}`;
  }

  private defaultLayer(type: LayerType, id: number): NetworkLayer {
    const map: Record<string, NetworkLayer> = {
      conv2d:     { id, type: 'conv2d',     name: `Conv ${id}`,       inputs: [], params: { outChannels: 8, kernelSize: 3, stride: 1, padding: 1, dilation: 1, kernelMatrix: [[0,-1,0],[-1,5,-1],[0,-1,0]], activation: 'relu' } },
      pool2d:     { id, type: 'pool2d',     name: `Pool ${id}`,       inputs: [], params: { mode: 'max', kernelSize: 2, stride: 2, padding: 0 } },
      residual:   { id, type: 'residual',   name: `Residual ${id}`,   inputs: [], params: { outChannels: 8, kernelSize: 3, stride: 1, padding: 1, activation: 'relu', useProjection: true } },
      flatten:    { id, type: 'flatten',    name: `Flatten ${id}`,    inputs: [], params: {} },
      dense:      { id, type: 'dense',      name: `Dense ${id}`,      inputs: [], params: { units: 64, activation: 'relu' } },
      activation: { id, type: 'activation', name: `Activation ${id}`, inputs: [], params: { activationType: 'relu' } },
      dropout:    { id, type: 'dropout',    name: `Dropout ${id}`,    inputs: [], params: { rate: 0.2, training: false } }
    };
    return map[type] ?? map['dense'];
  }
}
