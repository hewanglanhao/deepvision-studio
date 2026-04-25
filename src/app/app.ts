import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { Subscription } from 'rxjs';
import { HelpManualComponent } from './components/help-manual.component';
import { NetworkOverviewComponent } from './components/network-overview.component';
import { ForwardBackendService } from './services/forward-backend.service';
import { TrainingLog, TrainingRuntimeService } from './services/training-runtime.service';
import { SimEngine } from './sim-engine';
import {
  AppMode, Connection, DataSample, ExperimentResult,
  ConvKernelSpec,
  ForwardInputAsset, ForwardLayerResult, ForwardPassResult,
  ForwardTensor, InputLayer, LayerType, LayerValidationIssue, MetricPoint,
  ModelTemplate, NetworkLayer, PresetTask, TrainingConfig
} from './sim-models';

/** 上传图片显示预览最大边长（保留较高分辨率） */
const MAX_IMAGE_SIDE = 640;
/** 图片解码超时 ms */
const IMAGE_DECODE_TIMEOUT = 5000;
/** 像素网格按真实尺寸展示（不做前端缩略） */
const MAX_PREVIEW_GRID_SIDE = Number.POSITIVE_INFINITY;

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

@Component({
  selector: 'app-root',
  imports: [CommonModule, FormsModule, DecimalPipe, HelpManualComponent, NetworkOverviewComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnInit, OnDestroy {
  mode: AppMode = 'forward';
  showHelp = false;
  showSamplePicker = false;

  modelTemplates: ModelTemplate[] = SimEngine.templates();
  selectedTemplateId = 'cnn-classic';
  layers: NetworkLayer[] = [];
  connections: Connection[] = [];
  nextLayerId = 1;
  selectedLayerId = -1;

  datasets: Record<string, DataSample[]> = {
    MNIST: SimEngine.generateDataset(16, 'mnist'),
    'CIFAR-10': SimEngine.generateDataset(16, 'cifar')
  };
  selectedDataset = 'MNIST';
  selectedSampleId = 1;
  uploadComputeProfile: 'fast' | 'balanced' | 'quality' | 'original' = 'balanced';
  uploadedImageUrl = '';
  uploadError = '';
  private uploadedImageData: ImageData | null = null;
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
  trainingAcc = 0.2;
  trainingValAcc = 0.18;
  trainingHistory: MetricPoint[] = [];
  trainingLogs: TrainingLog[] = [];

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

  private subs = new Subscription();
  private tensorPreviewCache = new WeakMap<ForwardTensor, { mode: 'rgb' | 'gray'; width: number; height: number; colors?: string[]; values?: number[] }>();
  private rgbColorsCache = new WeakMap<object, string[]>();
  private tensorChannelPreviewCache = new WeakMap<ForwardTensor, ChannelPreviewItem[]>();
  private forwardDebounceTimer: number | null = null;
  private forwardRequestSeq = 0;
  private activeForwardAbort: AbortController | null = null;

  constructor(
    private trainingSvc: TrainingRuntimeService,
    private forwardBackend: ForwardBackendService
  ) {}

  ngOnInit(): void {
    this.applyTemplate();
    this.selectDataset('MNIST');
    this.subs.add(this.trainingSvc.state$.subscribe(s => {
      this.trainingStatus  = s.status;
      this.trainingEpoch   = s.currentEpoch;
      this.trainingLr      = s.currentLr;
      this.trainingLoss    = s.latestLoss;
      this.trainingAcc     = s.latestAccuracy;
      this.trainingValAcc  = s.latestValAccuracy;
    }));
    this.subs.add(this.trainingSvc.history$.subscribe(h => this.trainingHistory = h));
    this.subs.add(this.trainingSvc.logs$.subscribe(l => this.trainingLogs = l));
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    if (this.forwardDebounceTimer !== null) {
      window.clearTimeout(this.forwardDebounceTimer);
      this.forwardDebounceTimer = null;
    }
    this.activeForwardAbort?.abort();
    this.activeForwardAbort = null;
    this.trainingSvc.pause();
  }

  // ── Getters ──────────────────────────────────────────
  get layerCount() { return this.layers.length; }
  get parameterCount() { return SimEngine.parameterCount(this.layers, this.connections); }
  get layerPalette(): LayerType[] { return ['conv2d', 'pool2d', 'flatten', 'dense', 'activation', 'dropout']; }
  get selectedTemplate() { return this.modelTemplates.find(t => t.id === this.selectedTemplateId); }
  get selectedLayer() { return this.layers.find(l => l.id === this.selectedLayerId); }
  get inputLayer(): InputLayer | undefined { const l = this.layers.find(l => l.type === 'input'); return l?.type === 'input' ? l : undefined; }
  get datasetSamples() { return this.datasets[this.selectedDataset] ?? []; }
  get selectedSample() { return this.datasetSamples.find(s => s.id === this.selectedSampleId); }

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
    const [h, w, c] = t.shape as [number, number, number];
    const srcValues = t.values;
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
    const vals = viz.values;
    const n = (viz.width ?? 1) * (viz.height ?? 1);
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

  get isRgbInput(): boolean { return (this.currentInputAsset?.originalChannels ?? 1) >= 3; }
  get lossPolyline() { return SimEngine.buildPolyline(this.trainingHistory, 'loss'); }
  get accPolyline()  { return SimEngine.buildPolyline(this.trainingHistory, 'accuracy'); }
  get valPolyline()  { return SimEngine.buildPolyline(this.trainingHistory, 'valAccuracy'); }

  get validationIssues(): LayerValidationIssue[] { return this.forwardResult?.validationIssues ?? []; }

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
    for (const err of this.forwardResult?.errors ?? []) {
      const layerName = err.split(':')[0]?.trim();
      const layer = this.layers.find(l => l.name === layerName);
      if (layer) ids.add(layer.id);
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
    if (m === 'forward') { this.trainingSvc.pause(); this.runForward(); }
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
    this.selectedSampleId = this.datasetSamples[0]?.id ?? 1;
    this.uploadedImageUrl = ''; this.uploadedImageData = null; this.uploadError = '';
    this.showSamplePicker = false;
    this.rebuildInputAsset(); this.runForward();
  }

  chooseSample(id: number): void {
    this.selectedSampleId = id;
    this.uploadedImageUrl = ''; this.uploadedImageData = null; this.uploadError = '';
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

  // ── Forward pass ──────────────────────────────────────
  runForward(force = false): void {
    if (this.mode !== 'forward') return;
    if (!force && !this.autoForwardCompute) {
      this.pendingForwardChanges = true;
      this.forwardStatusMessage = '参数已更新，点击“开始计算”执行前向传播。';
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
      this.pendingForwardChanges = false;
      this.forwardStatusMessage = '计算中...';
      this.activeForwardAbort?.abort();
      this.activeForwardAbort = new AbortController();
      try {
        const remote = await this.forwardBackend.executeForward({
          layers: this.layers,
          connections: this.connections,
          inputTensor
        }, this.activeForwardAbort.signal);
        this.forwardBackendError = '';
        this.forwardStatusMessage = '计算完成。';
        this.applyForwardResult(remote, activeSeq);
      } catch (err) {
        if (err instanceof DOMException && err.name === 'AbortError') {
          this.forwardStatusMessage = '计算已取消。';
          return;
        }
        this.forwardBackendError = '后端不可用，前端本地前向推理已移除。';
        this.forwardStatusMessage = '后端请求失败。';
      } finally {
        if (activeSeq === this.forwardRequestSeq) {
          this.forwardBusy = false;
          this.activeForwardAbort = null;
        }
      }
    }, 80);
  }

  triggerForwardCompute(): void {
    this.runForward(true);
  }

  cancelForwardCompute(): void {
    this.activeForwardAbort?.abort();
    this.activeForwardAbort = null;
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

  // ── Training ──────────────────────────────────────────
  startTraining(): void {
    this.trainingSvc.configure(this.trainingConfig, this.layers);
    this.trainingSvc.start();
  }
  pauseTraining(): void  { this.trainingSvc.pause(); }
  stopTraining(): void   { this.trainingSvc.stop(); }
  resetTraining(): void  { this.trainingSvc.reset(); }
  selectTask(id: string): void { this.selectedTaskId = id; }

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
    if (!this.uploadedImageData) return;
    this.applyUploadComputeProfile(this.uploadedImageData.width, this.uploadedImageData.height);
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
    } else {
      const s = this.selectedSample;
      if (!s) { this.currentInputAsset = null; return; }
      this.currentInputAsset = SimEngine.createForwardInputAssetFromSample(s, pre);
    }
    this.syncInputShape();
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
    const [h, w, c] = tensor.shape as [number, number, number];
    const srcValues = tensor.values;
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

  private defaultLayer(type: LayerType, id: number): NetworkLayer {
    const map: Record<string, NetworkLayer> = {
      conv2d:     { id, type: 'conv2d',     name: `Conv ${id}`,       inputs: [], params: { outChannels: 8, kernelSize: 3, stride: 1, padding: 1, dilation: 1, kernelMatrix: [[0,-1,0],[-1,5,-1],[0,-1,0]], activation: 'relu' } },
      pool2d:     { id, type: 'pool2d',     name: `Pool ${id}`,       inputs: [], params: { mode: 'max', kernelSize: 2, stride: 2, padding: 0 } },
      flatten:    { id, type: 'flatten',    name: `Flatten ${id}`,    inputs: [], params: {} },
      dense:      { id, type: 'dense',      name: `Dense ${id}`,      inputs: [], params: { units: 64, activation: 'relu' } },
      activation: { id, type: 'activation', name: `Activation ${id}`, inputs: [], params: { activationType: 'relu' } },
      dropout:    { id, type: 'dropout',    name: `Dropout ${id}`,    inputs: [], params: { rate: 0.2, training: false } }
    };
    return map[type] ?? map['dense'];
  }
}
