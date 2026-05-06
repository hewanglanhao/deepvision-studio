import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { Subscription } from 'rxjs';
import { HelpManualComponent } from '../../components/help-manual.component';
import { NetworkOverviewComponent } from '../../components/network-overview.component';
import { NETWORK_3D_SESSION_KEY, Network3dPayload } from '../../features/network-3d/network-3d.models';
import { LlmFloatingAssistantComponent } from '../../features/llm/llm-floating-assistant.component';
import { LlmChatContext } from '../../features/llm/llm.models';
import { MODE_A_LLM_SYSTEM_PROMPT } from '../../features/llm/llm-prompts';
import { AuthUser } from '../../models/auth.models';
import { ForwardRecordDetail, ForwardRecordSummary, ForwardRecordSnapshot } from '../../models/forward-record.models';
import { AuthService } from '../../services/auth.service';
import { ForwardRecordService } from '../../services/forward-record.service';
import { ForwardBackendService } from '../../services/forward-backend.service';
import { SimEngine } from '../../sim-engine';
import {
  AppMode, Connection, DataSample,
  ConvKernelSpec,
  ForwardInputAsset, ForwardLayerResult, ForwardPassResult,
  ForwardTensor, InputLayer, LayerType,
  LayerValidationIssue, ModelTemplate, NetworkLayer, TensorShape
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

interface LayerFormulaView {
  title: string;
  expressionHtml: string;
  detail: string;
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

type EditableBiasParams = { bias?: number[] };

@Component({
  selector: 'app-mode-a-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    DecimalPipe,
    RouterModule,
    HelpManualComponent,
    NetworkOverviewComponent,
    LlmFloatingAssistantComponent
  ],
  templateUrl: './mode-a-page.component.html',
  styleUrl: './mode-a-page.component.css'
})
export class ModeAPageComponent implements OnInit, OnDestroy {
  readonly modeALlmSystemPrompt = MODE_A_LLM_SYSTEM_PROMPT;
  readonly modeALlmContextProvider = (): LlmChatContext => this.buildModeALlmContext();

  mode: AppMode = 'forward';
  showHelp = false;
  showSamplePicker = false;
  authUser: AuthUser | null = null;
  showAuthModal = false;
  authMode: 'login' | 'register' = 'login';
  authDraft = { username: '', password: '', displayName: '' };
  authBusy = false;
  authError = '';

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

  readonly kernelPresets = KERNEL_PRESETS;
  selectedKernelOutChannel = 0;
  selectedKernelInChannel = 0;
  showChannelModal = false;
  channelModalTitle = '';
  channelModalPreviews: ChannelPreviewItem[] = [];

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
  selectedDenseWeightRows: Record<number, number> = {};

  constructor(
    private route: ActivatedRoute,
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
    this.subs.add(this.authSvc.user$.subscribe(user => {
      this.authUser = user;
      if (user && this.showRecordDrawer) {
        this.loadForwardRecords();
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
  }

  // ── Getters ──────────────────────────────────────────
  get layerCount() { return this.layers.length; }
  get parameterCount() { return SimEngine.parameterCount(this.layers, this.connections); }
  get layerPalette(): LayerType[] { return ['conv2d', 'pool2d', 'flatten', 'dense', 'activation', 'dropout']; }
  get selectedTemplate() { return this.modelTemplates.find(t => t.id === this.selectedTemplateId); }
  get selectedLayer() { return this.layers.find(l => l.id === this.selectedLayerId); }
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
  get selectedForwardResult(): ForwardLayerResult | null {
    if (!this.forwardResult?.layerResults.length) return null;
    return this.forwardResult.layerResults.find(r => r.layerId === this.selectedLayerId)
      ?? this.forwardResult.layerResults[0];
  }

  get selectedFormula(): LayerFormulaView | null {
    const result = this.selectedForwardResult;
    if (!result) return null;
    const layer = this.layers.find(l => l.id === result.layerId);
    return this.buildLayerFormula(result, layer);
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

  openNetwork3dViewer(): void {
    const payload: Network3dPayload = {
      title: 'A 模式网络层 3D 展示',
      sourceMode: 'Mode A',
      createdAt: new Date().toISOString(),
      inputImageUrl: this.preparedInputImageUrl || this.originalInputImageUrl,
      inputLabel: this.currentInputAsset?.name,
      layers: structuredClone(this.layers),
      shapeHints: structuredClone(this.forwardLayerShapeMap),
      layerShapes: this.buildNetwork3dLayerShapes(),
      selectedLayerId: this.selectedLayerId
    };

    localStorage.setItem(NETWORK_3D_SESSION_KEY, JSON.stringify(payload));
    window.open('/network-3d', '_blank', 'noopener,noreferrer');
  }

  private buildModeALlmContext(): LlmChatContext {
    const selectedResult = this.selectedForwardResult;
    const selectedInputTensor = this.selectedLayerInputTensor();
    const selectedOutputTensor = selectedResult?.tensor ?? null;
    const lines: string[] = [
      `模式: A 模式 / 图片前向传播可视化`,
      `数据集: ${this.selectedDataset}`,
      `当前样本: ${this.currentInputAsset?.name ?? '未选择'}${this.currentInputAsset?.label ? ` / 标签 ${this.currentInputAsset.label}` : ''}`,
      `网络层数: ${this.layerCount}`,
      `参数量估计: ${this.parameterCount}`,
      `当前选中层: ${this.selectedLayer?.name ?? '无'} (${this.selectedLayer?.type ?? '-'})`,
      `是否存在后端前向结果: ${this.forwardResult ? '是' : '否'}`
    ];

    if (this.currentInputAsset) {
      lines.push(
        `输入原始尺寸: ${this.currentInputAsset.originalWidth}x${this.currentInputAsset.originalHeight}x${this.currentInputAsset.originalChannels}`,
        `实际计算尺寸: ${this.currentInputAsset.prepared.tensor.shape.join('x')}`,
        `预处理备注: ${this.currentInputAsset.prepared.notes.join(', ') || '无'}`
      );
    }

    lines.push('', '网络结构:');
    for (const [index, layer] of this.layers.entries()) {
      lines.push(`${index + 1}. ${layer.name} / ${layer.type} / inputs=${layer.inputs.join(',') || 'none'} / params=${this.safeJson(layer.params)}`);
    }

    if (this.forwardResult?.shapePath?.length) {
      lines.push('', `Shape path: ${this.forwardResult.shapePath.join(' -> ')}`);
    }

    if (selectedResult) {
      lines.push(
        '',
        '当前选中层前向结果:',
        `层名: ${selectedResult.layerName}`,
        `类型: ${selectedResult.layerType}`,
        `输入形状: ${selectedResult.inputShapes.map(shape => `[${shape.join(', ')}]`).join(', ') || '无'}`,
        `输出形状: ${selectedResult.outputShapeLabel}`,
        `输入张量摘要: ${this.tensorSummary(selectedInputTensor)}`,
        `输出张量摘要: ${this.tensorSummary(selectedOutputTensor)}`,
        `转换说明: ${selectedResult.transitionNote}`,
        `参数摘要: ${selectedResult.paramsSummary.join('; ') || '无'}`,
        `警告: ${selectedResult.warnings.join('; ') || '无'}`,
        `统计: min=${selectedResult.stats.min.toFixed(4)}, max=${selectedResult.stats.max.toFixed(4)}, mean=${selectedResult.stats.mean.toFixed(4)}, nonZeroRatio=${selectedResult.stats.nonZeroRatio.toFixed(4)}`
      );
    }

    if (this.selectedConvLayer) {
      lines.push(
        '',
        '当前卷积核:',
        `Out ${this.selectedKernelOutChannel} / In ${this.selectedKernelInChannel}`,
        this.editableKernelMatrix.map(row => row.map(value => Number(value).toFixed(3)).join('\t')).join('\n')
      );
    }

    if (selectedResult?.stats.topK?.length) {
      lines.push(
        '',
        `当前层 Top-K/最大值: ${selectedResult.stats.topK.map(item => `${item.label ?? item.index}: ${item.value.toFixed(4)}`).join(', ')}`
      );
    }

    const images: Array<{ title: string; url: string }> = [];
    const inputImageUrl = this.tensorImageUrlForContext(selectedInputTensor);
    if (inputImageUrl) {
      images.push({ title: `${selectedResult?.layerName ?? '当前层'} 的输入图像`, url: inputImageUrl });
    }
    const outputImageUrl = this.tensorImageUrlForContext(selectedOutputTensor);
    if (outputImageUrl) {
      images.push({ title: `${selectedResult?.layerName ?? '当前层'} 的输出图像`, url: outputImageUrl });
    }

    return {
      text: lines.join('\n'),
      images
    };
  }

  private selectedLayerInputTensor(): ForwardTensor | null {
    const selected = this.selectedForwardResult;
    if (!selected) return null;
    if (selected.layerType === 'input') {
      return this.currentInputAsset?.prepared.tensor ?? null;
    }
    const layerIndex = this.forwardResult?.layerResults.findIndex(result => result.layerId === selected.layerId) ?? -1;
    if (layerIndex > 0) {
      return this.forwardResult?.layerResults[layerIndex - 1]?.tensor ?? null;
    }
    return null;
  }

  private tensorImageUrlForContext(tensor: ForwardTensor | null): string {
    if (!tensor || tensor.shape.length !== 3) return '';
    return this.tensorToImageDataUrl(tensor, !(tensor.shape[2] === 3 && tensor.colorMode === 'rgb'));
  }

  private tensorSummary(tensor: ForwardTensor | null): string {
    if (!tensor) return '无';
    const values = tensor.values ?? [];
    const preview = values.slice(0, 12).map(value => Number(value).toFixed(4)).join(', ');
    return `kind=${tensor.kind}, shape=[${tensor.shape.join(', ')}], values[0..${Math.min(values.length, 12)}]=${preview}${values.length > 12 ? '...' : ''}`;
  }

  convBiasValue(layer: Extract<NetworkLayer, { type: 'conv2d' }>): number {
    return layer.params.bias?.[this.selectedKernelOutChannel] ?? 0;
  }

  onConvBiasInput(layer: Extract<NetworkLayer, { type: 'conv2d' }>, value: string | number): void {
    const outChannels = Math.max(1, layer.params.outChannels);
    const bias = layer.params.bias ?? Array.from({ length: outChannels }, () => 0);
    while (bias.length < outChannels) bias.push(0);
    bias[this.selectedKernelOutChannel] = this.finiteNumber(value, bias[this.selectedKernelOutChannel] ?? 0);
    layer.params.bias = bias.slice(0, outChannels);
    this.runForward();
  }

  denseUnitIndices(layer: NetworkLayer): number[] {
    if (layer.type !== 'dense' && layer.type !== 'output') return [];
    return Array.from({ length: Math.max(1, layer.params.units) }, (_, index) => index);
  }

  selectedDenseWeightRow(layer: NetworkLayer): number {
    if (layer.type !== 'dense' && layer.type !== 'output') return 0;
    const max = Math.max(1, layer.params.units) - 1;
    const current = this.selectedDenseWeightRows[layer.id] ?? 0;
    const next = Math.max(0, Math.min(max, current));
    this.selectedDenseWeightRows[layer.id] = next;
    return next;
  }

  onDenseWeightRowChange(layer: NetworkLayer, row: number): void {
    if (layer.type !== 'dense' && layer.type !== 'output') return;
    const max = Math.max(1, layer.params.units) - 1;
    this.selectedDenseWeightRows[layer.id] = Math.max(0, Math.min(max, Number(row) || 0));
  }

  denseBiasValue(layer: NetworkLayer, row: number): number {
    if (layer.type !== 'dense' && layer.type !== 'output') return 0;
    return layer.params.bias?.[row] ?? 0;
  }

  onDenseBiasInput(layer: NetworkLayer, value: string | number): void {
    if (layer.type !== 'dense' && layer.type !== 'output') return;
    const units = Math.max(1, layer.params.units);
    const row = this.selectedDenseWeightRow(layer);
    const params = layer.params as EditableBiasParams;
    const bias = params.bias ?? Array.from({ length: units }, () => 0);
    while (bias.length < units) bias.push(0);
    bias[row] = this.finiteNumber(value, bias[row] ?? 0);
    params.bias = bias.slice(0, units);
    this.runForward();
  }

  private finiteNumber(value: string | number, fallback: number): number {
    const next = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(next) ? next : fallback;
  }

  private buildLayerFormula(result: ForwardLayerResult, layer?: NetworkLayer): LayerFormulaView {
    const params = layer?.params as Record<string, any> | undefined;
    const input = result.inputShapes[0] ?? [];
    const output = result.outputShape ?? [];
    if (result.layerType === 'conv2d') {
      const k = this.numParam(params, 'kernelSize', 3);
      const stride = this.numParam(params, 'stride', 1);
      const pad = this.numParam(params, 'padding', 0);
      const dilation = this.numParam(params, 'dilation', 1);
      const outChannels = this.numParam(params, 'outChannels', output[2] ?? 1);
      const h = input[0] ?? '?';
      const w = input[1] ?? '?';
      return {
        title: '卷积层计算',
        expressionHtml: '<i>Y</i><sub>o,y,x</sub> = <span class="sigma">Σ</span><sub>c,i,j</sub> <i>X</i><sub>c,y+i,x+j</sub><i>K</i><sub>o,c,i,j</sub> + <i>b</i><sub>o</sub>',
        detail: `空间尺寸按 floor((输入 + 2P - D*(K-1) - 1) / S + 1) 计算；每个输出通道使用一组卷积核在输入特征图上滑动求和。当前输入 ${h}x${w}，输出 ${result.outputShapeLabel}。`
      };
    }

    if (result.layerType === 'pool2d') {
      const k = this.numParam(params, 'kernelSize', 2);
      const stride = this.numParam(params, 'stride', k);
      const pad = this.numParam(params, 'padding', 0);
      const mode = String(params?.['mode'] ?? 'max');
      return {
        title: mode === 'avg' ? '平均池化计算' : '最大池化计算',
        expressionHtml: mode === 'avg'
          ? '<i>Y</i><sub>y,x,c</sub> = mean(window(<i>X</i>))'
          : '<i>Y</i><sub>y,x,c</sub> = max(window(<i>X</i>))',
        detail: `池化不学习参数，只在每个通道内用 ${k}x${k} 窗口压缩空间尺寸；shape 仍按卷积类窗口公式计算。`
      };
    }

    if (result.layerType === 'flatten') {
      const size = input.reduce((acc, v) => acc * v, 1);
      return {
        title: '展平计算',
        expressionHtml: '<i>Y</i><sub>index</sub> = reshape(<i>X</i><sub>h,w,c</sub>)',
        detail: `Flatten 只改变张量排列方式，不改变数值本身；${result.inputShapeLabel} 被拉平成长度 ${Number.isFinite(size) ? size : result.outputShapeLabel} 的向量。`
      };
    }

    if (result.layerType === 'dense' || result.layerType === 'output') {
      const units = this.numParam(params, 'units', output[0] ?? 1);
      const activation = String(params?.['activation'] ?? 'none');
      return {
        title: result.layerType === 'output' ? '输出层计算' : '全连接层计算',
        expressionHtml: activation && activation !== 'none'
          ? '<i>Y</i> = activation(<i>W</i><i>x</i> + <i>b</i>)'
          : '<i>Y</i> = <i>W</i><i>x</i> + <i>b</i>',
        detail: 'Dense/Output 使用真实矩阵乘法完成前向传播。当前模式不做训练，权重由系统按层 ID 生成确定性演示权重，偏置可在左侧手动设置。'
      };
    }

    if (result.layerType === 'activation') {
      const activation = String(params?.['activationType'] ?? params?.['activation'] ?? 'relu');
      const expressionHtml = activation === 'relu'
        ? '<i>Y</i> = max(0, <i>X</i>)'
        : activation === 'sigmoid'
          ? '<i>Y</i> = 1 / (1 + exp(-<i>X</i>))'
          : activation === 'tanh'
            ? '<i>Y</i> = tanh(<i>X</i>)'
            : activation === 'softmax'
              ? '<i>Y</i><sub>i</sub> = exp(<i>X</i><sub>i</sub>) / <span class="sigma">Σ</span><sub>j</sub> exp(<i>X</i><sub>j</sub>)'
              : `<i>Y</i> = ${activation}(<i>X</i>)`;
      return {
        title: '激活函数计算',
        expressionHtml,
        detail: '激活层逐元素改变数值分布，通常不改变 shape；它负责引入非线性，让网络不只是线性变换的叠加。'
      };
    }

    if (result.layerType === 'dropout') {
      const rate = this.numParam(params, 'rate', 0);
      const enabled = !!params?.['training'];
      return {
        title: 'Dropout 前向演示',
        expressionHtml: enabled
          ? '<i>Y</i> = <i>X</i> · mask / (1 - rate)'
          : '<i>Y</i> = <i>X</i>',
        detail: enabled
          ? '当前启用了随机丢弃演示，会按比例遮蔽部分激活值；这是前向传播中的随机算子展示，不代表 A 模式在训练模型。'
          : '当前未启用随机丢弃，Dropout 在前向传播中保持输入不变。'
      };
    }

    return {
      title: '输入层',
      expressionHtml: '<i>Y</i> = <i>X</i>',
      detail: '输入层负责把预处理后的图片张量送入网络，不改变数值。'
    };
  }

  private numParam(params: Record<string, any> | undefined, key: string, fallback: number): number {
    const value = Number(params?.[key]);
    return Number.isFinite(value) ? value : fallback;
  }

  private safeJson(value: unknown): string {
    try {
      const text = JSON.stringify(value);
      return text.length > 900 ? `${text.slice(0, 900)}...` : text;
    } catch {
      return String(value);
    }
  }

  get isRgbInput(): boolean { return (this.currentInputAsset?.originalChannels ?? 1) >= 3; }
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

  private buildNetwork3dLayerShapes(): Record<number, TensorShape> {
    const shapes: Record<number, TensorShape> = {};
    for (const result of this.forwardResult?.layerResults ?? []) {
      shapes[result.layerId] = result.outputShape;
    }
    if (Object.keys(shapes).length) {
      return shapes;
    }

    for (const layer of this.layers) {
      const inputShapes = (layer.inputs ?? [])
        .map((id) => shapes[id])
        .filter((shape): shape is TensorShape => shape !== undefined);
      shapes[layer.id] = SimEngine.inferLayerOutputShape(layer, inputShapes);
    }
    return shapes;
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
