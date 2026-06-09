import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, RouterModule } from '@angular/router';
import { Subscription } from 'rxjs';
import { NetworkOverviewComponent } from '@shared/network/network-overview.component';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import { NETWORK_3D_SESSION_KEY, Network3dLayerSnapshot, Network3dPayload } from '@shared/network-3d/network-3d.models';
import { LlmFloatingAssistantComponent, LlmQuickPrompt } from '@shared/llm/llm-floating-assistant.component';
import { TeachingSearchFabComponent } from '@shared/teaching/teaching-search-fab.component';
import { TeachingTermDirective } from '@shared/teaching/teaching-term.directive';
import { LlmChatContext } from '@shared/llm/llm.models';
import { MODE_A_LLM_SYSTEM_PROMPT } from '@shared/llm/llm-prompts';
import { AuthUser } from '@core/auth/auth.models';
import { ForwardRecordDetail, ForwardRecordSummary, ForwardRecordSnapshot } from '@shared/forward/forward-record.models';
import { AuthService } from '@core/auth/auth.service';
import { ForwardRecordService } from '@shared/forward/forward-record.service';
import { ForwardBackendService } from '@shared/forward/forward-backend.service';
import { SimEngine } from '@shared/simulation/sim-engine';
import {
  AppMode, Connection, DataSample,
  ConvKernelSpec,
  ForwardInputAsset, ForwardLayerResult, ForwardPassResult,
  ForwardTensor, InputLayer, LayerType,
  LayerValidationIssue, ModelTemplate, NetworkLayer, TensorShape, TensorStats
} from '@shared/simulation/sim-models';

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

interface KernelCompareItem {
  label: string;
  matrix: number[][];
  imageUrl: string;
  outputShapeLabel: string;
  stats: TensorStats;
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
    NetworkOverviewComponent,
    PlatformTopbarComponent,
    LlmFloatingAssistantComponent,
    TeachingSearchFabComponent,
    TeachingTermDirective
  ],
  templateUrl: './mode-a-page.component.html',
  styleUrl: './mode-a-page.component.css'
})
export class ModeAPageComponent implements OnInit, OnDestroy {
  readonly modeALlmSystemPrompt = MODE_A_LLM_SYSTEM_PROMPT;
  readonly modeALlmContextProvider = (): LlmChatContext => this.buildModeALlmContext();
  readonly modeALlmQuickPrompts: LlmQuickPrompt[] = [
    {
      label: '解释当前层',
      question: '请结合当前 A 模式页面数据，解释当前选中层的输入、输出、公式和可视化结果。'
    },
    {
      label: '卷积核差异',
      question: '请对比当前卷积核和卷积核对比面板中的 Identity、Edge、Sharpen、Blur、Sobel 结果，说明它们为什么产生不同特征图。'
    },
    {
      label: '输出形状',
      question: '请逐步说明当前选中层的输出 shape 是如何由输入 shape 和层参数计算出来的。'
    },
    {
      label: '公式讲解',
      question: '请把当前选中层的前向传播公式用初学者能听懂的话解释一遍，并说明公式里每个符号代表什么。'
    },
    {
      label: '配置诊断',
      question: '请检查当前网络结构和参数，指出可能导致输出不直观、维度不合理或特征图异常的配置。'
    },
    {
      label: '答辩总结',
      question: '请用 1 分钟答辩口吻总结 A 模式展示了什么，并说明它为什么是真实前向传播演示而不是训练模拟。'
    }
  ];

  mode: AppMode = 'forward';
  showSamplePicker = false;
  authUser: AuthUser | null = null;
  showAuthModal = false;
  authMode: 'login' | 'register' = 'login';
  authDraft = { username: '', password: '', displayName: '' };
  authBusy = false;
  authError = '';

  /** 顶部状态展示当前网络层数和参数量，用来快速判断这个前向传播模型的复杂度。 */
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

  readonly kernelPresets = KERNEL_PRESETS;
  selectedKernelOutChannel = 0;
  selectedKernelInChannel = 0;
  showChannelModal = false;
  channelModalTitle = '';
  channelModalPreviews: ChannelPreviewItem[] = [];
  showKernelCompareModal = false;
  kernelCompareBusy = false;
  kernelCompareError = '';
  kernelCompareItems: KernelCompareItem[] = [];
  private kernelCompareRequestSeq = 0;

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

  /** 注入路由、forward 后端、认证和历史记录服务，A 模式的计算、登录和快照都依赖这些服务。 */
  constructor(
    private route: ActivatedRoute,
    private forwardBackend: ForwardBackendService,
    private authSvc: AuthService,
    private forwardRecordSvc: ForwardRecordService
  ) {}

  /** 初始化页面状态、订阅数据源并触发首次数据加载。 */
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

  /** 释放组件订阅、定时器和渲染资源，避免页面离开后继续占用内存。 */
  ngOnDestroy(): void {
    this.subs.unsubscribe();
    if (this.forwardDebounceTimer !== null) {
      window.clearTimeout(this.forwardDebounceTimer);
      this.forwardDebounceTimer = null;
    }
  }

  // ── Getters ──────────────────────────────────────────
  /** 返回当前网络层数量，供顶部状态和记录摘要展示。 */
  get layerCount() { return this.layers.length; }
  /** 估算当前网络的可学习参数量，卷积核和 Dense 权重越多，模型容量和计算量越大。 */
  get parameterCount() { return SimEngine.parameterCount(this.layers, this.connections); }
  /** 返回允许用户新增的层类型，覆盖 CNN 前向传播常见的卷积、池化、Flatten、Dense 和激活层。 */
  get layerPalette(): LayerType[] { return ['conv2d', 'pool2d', 'flatten', 'dense', 'activation', 'dropout']; }
  /** 找到当前选中的模型模板，例如 MLP、经典 CNN 或残差 CNN。 */
  get selectedTemplate() { return this.modelTemplates.find(t => t.id === this.selectedTemplateId); }
  /** 找到右侧检查器正在查看的网络层，后续公式、参数和特征图都围绕它展示。 */
  get selectedLayer() { return this.layers.find(l => l.id === this.selectedLayerId); }
  /** 获取输入层；输入层决定图片张量的 height、width、channels 和预处理方式。 */
  get inputLayer(): InputLayer | undefined { const l = this.layers.find(l => l.type === 'input'); return l?.type === 'input' ? l : undefined; }
  /** 获取输出层；输出层通常把最后的隐藏向量映射成类别 logits 或 Softmax 概率。 */
  get outputLayer() { const l = this.layers.find(l => l.type === 'output'); return l?.type === 'output' ? l : undefined; }
  /** 返回当前数据集的内置样本列表，用于观察同一网络在不同图片上的特征响应。 */
  get datasetSamples() { return this.datasets[this.selectedDataset] ?? []; }
  /** 找到当前选中的演示样本，它会被转成输入张量送入 forward。 */
  get selectedSample() { return this.datasetSamples.find(s => s.id === this.selectedSampleId); }
  /** 汇总本地图像样本的类别名称，给样本选择器按数据集分组。 */
  get localDatasetNames(): string[] {
    return [...new Set(this.localImageSamples.map(sample => sample.category))];
  }
  /** 返回当前数据集下的本地图像样本，避免动物、花朵等类别混在同一个选择面板里。 */
  get activeLocalImageSamples(): LocalImageSample[] {
    return this.localImageSamples.filter(sample => sample.category === this.selectedDataset);
  }
  /** 找到当前选中的本地图像样本，后续会解码为 ImageData 再转成 CNN 输入张量。 */
  get selectedLocalImageSample(): LocalImageSample | undefined {
    return this.localImageSamples.find(sample => sample.id === this.selectedLocalImageId);
  }
  /** 统计某个本地数据集有多少样本，用于样本选择器展示数量。 */
  localDatasetSampleCount(dataset: string): number {
    return this.localImageSamples.filter(sample => sample.category === dataset).length;
  }
  /** 获取当前选中层的真实 forward 结果，包括输出 shape、张量值、统计量和可视化摘要。 */
  get selectedForwardResult(): ForwardLayerResult | null {
    if (!this.forwardResult?.layerResults.length) return null;
    return this.forwardResult.layerResults.find(r => r.layerId === this.selectedLayerId)
      ?? this.forwardResult.layerResults[0];
  }

  /** 为当前层生成公式说明，例如卷积输出尺寸、池化窗口、Dense 的 W·x+b 或 Softmax。 */
  get selectedFormula(): LayerFormulaView | null {
    const result = this.selectedForwardResult;
    if (!result) return null;
    const layer = this.layers.find(l => l.id === result.layerId);
    return this.buildLayerFormula(result, layer);
  }

  /** 把当前层向量响应归一化成条形图高度，用于观察神经元激活强弱。 */
  get selectedBars(): number[] {
    return this.normBars((this.selectedForwardResult?.visualization.values ?? []).slice(0, 64));
  }

  /** 根据原始图片通道数给出颜色模式选项；RGB/灰度会改变第一层卷积核的输入通道数。 */
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

  /** 判断当前选中层是否为卷积层；只有卷积层才显示 kernel 编辑和通道对比工具。 */
  get selectedConvLayer() {
    const layer = this.selectedLayer;
    return layer?.type === 'conv2d' ? layer : null;
  }

  /** 计算当前卷积层的输入通道数；多通道卷积会为每个输入通道准备一张 kernel 矩阵。 */
  get selectedConvInChannels(): number {
    const layer = this.selectedConvLayer;
    if (!layer) return 1;
    const result = this.forwardResult?.layerResults.find(r => r.layerId === layer.id);
    const shape = result?.inputShapes?.[0];
    if (shape && shape.length === 3) return Math.max(1, shape[2]);
    return Math.max(1, this.inputLayer?.params.channels ?? 1);
  }

  /** 返回卷积输出通道索引，每个输出通道可理解为一个卷积核提取到的一类特征图。 */
  get convOutChannelIndices(): number[] {
    const out = this.selectedConvLayer?.params.outChannels ?? 1;
    return Array.from({ length: Math.max(1, out) }, (_, i) => i);
  }

  /** 返回卷积输入通道索引，用于查看 RGB 或上游多通道特征图对应的 kernel 权重。 */
  get convInChannelIndices(): number[] {
    const inC = this.selectedConvInChannels;
    return Array.from({ length: Math.max(1, inC) }, (_, i) => i);
  }

  /** 只有输入特征图有多个通道时才显示输入通道选择器。 */
  get showConvInChannelSelector(): boolean {
    return this.selectedConvInChannels > 1;
  }

  /** 取出当前正在编辑的卷积核矩阵；矩阵数值决定边缘、模糊、锐化等局部特征响应。 */
  get editableKernelMatrix(): number[][] {
    const layer = this.selectedConvLayer;
    if (!layer) return [];
    this.ensureConvKernelBank(layer);
    return layer.params.kernels?.[this.selectedKernelOutChannel]?.weights?.[this.selectedKernelInChannel]
      ?? layer.params.kernelMatrix
      ?? [];
  }

  /** 判断最终输出是图像特征图还是向量；分类网络最后通常是向量概率分布。 */
  get finalTensorMode(): 'image' | 'vector' | 'none' {
    const shapeLen = this.forwardResult?.finalTensor?.shape.length ?? 0;
    if (shapeLen === 3) return 'image';
    if (shapeLen >= 1) return 'vector';
    return 'none';
  }

  /** 将最终输出向量转成条形图，常用于展示各类别 logits/概率的相对大小。 */
  get finalBars(): number[] {
    if (this.finalTensorMode !== 'vector') return [];
    return this.normBars((this.forwardResult?.finalTensor?.values ?? []).slice(0, 32));
  }

  /** 当最终输出仍是三维特征图时，生成灰度/RGB 预览和通道预览。 */
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

  /** 取当前层前几个通道的特征图预览，帮助观察不同卷积核关注的局部模式。 */
  get selectedChannelPreviews(): ChannelPreviewItem[] {
    return (this.selectedForwardResult?.visualization.channelPreviews ?? []).slice(0, 4);
  }

  /** 返回当前层输出特征图的通道数，卷积层的通道数通常等于 outChannels。 */
  get selectedChannelCount(): number {
    const tensor = this.selectedForwardResult?.tensor;
    return tensor && tensor.shape.length === 3 ? tensor.shape[2] : 0;
  }

  /** 返回最终输出特征图的通道数，用于决定是否需要通道展开弹窗。 */
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

  /** 返回原始输入图的预览 URL，用来对比预处理前后图像是否发生尺寸或通道变化。 */
  get originalInputImageUrl(): string {
    return this.currentInputAsset?.previewUrl
      || this.tensorToImageDataUrl(this.currentInputAsset?.originalTensor ?? null);
  }

  /** 返回真正送入网络计算的预处理输入图，forward 使用的是这个张量而不是原图。 */
  get preparedInputImageUrl(): string {
    return this.tensorToImageDataUrl(this.currentInputAsset?.prepared.displayTensor ?? null);
  }

  /** 把当前层输出张量转成图片 URL，方便直接观察卷积/池化后的特征图。 */
  get selectedTensorImageUrl(): string {
    const tensor = this.selectedForwardResult?.tensor ?? null;
    return this.tensorToImageDataUrl(tensor, !(tensor?.shape.length === 3 && tensor.shape[2] === 3 && tensor.colorMode === 'rgb'));
  }

  /** 把最终输出张量转成图片 URL；如果最后仍是特征图，就可以继续作为图像查看。 */
  get finalTensorImageUrl(): string {
    const tensor = this.forwardResult?.finalTensor ?? null;
    return this.tensorToImageDataUrl(tensor, !(tensor?.shape.length === 3 && tensor.shape[2] === 3 && tensor.colorMode === 'rgb'));
  }

  /** 把单个通道的归一化特征响应转成灰度图 URL。 */
  channelPreviewImageUrl(channel: ChannelPreviewItem): string {
    const cached = this.channelImagePreviewCache.get(channel);
    if (cached) return cached;
    const url = this.grayValuesToImageDataUrl(channel.values, channel.width, channel.height);
    this.channelImagePreviewCache.set(channel, url);
    return url;
  }

  /** 打开大图查看器，用于放大输入图、特征图或卷积通道响应。 */
  openImageViewer(title: string, url: string, meta = ''): void {
    if (!url) return;
    this.imageViewer = { open: true, title, url, meta };
  }

  /** 关闭大图查看器并清理当前展示的图像信息。 */
  closeImageViewer(): void {
    this.imageViewer = { open: false, title: '', url: '', meta: '' };
  }

  /** 把当前网络快照写入 sessionStorage 并打开 3D 查看页。 */
  openNetwork3dViewer(): void {
    const payload: Network3dPayload = {
      title: 'A 模式网络层 3D 展示',
      sourceMode: 'Mode A',
      createdAt: new Date().toISOString(),
      inputImageUrl: this.preparedInputImageUrl || this.originalInputImageUrl,
      inputLabel: this.currentInputAsset?.name,
      datasetName: this.selectedDataset,
      parameterCount: this.parameterCount,
      layers: structuredClone(this.layers),
      shapeHints: structuredClone(this.forwardLayerShapeMap),
      layerShapes: this.buildNetwork3dLayerShapes(),
      layerSnapshots: this.buildNetwork3dLayerSnapshots(),
      shapePath: structuredClone(this.forwardResult?.shapePath ?? []),
      finalTopK: structuredClone(this.forwardResult?.finalTopK ?? []),
      selectedLayerId: this.selectedLayerId
    };

    localStorage.setItem(NETWORK_3D_SESSION_KEY, JSON.stringify(payload));
    window.open('/network-3d', '_blank', 'noopener,noreferrer');
  }

  /** 整理当前 A 模式页面上下文，供 LLM 助手解释实验状态。 */
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

  /** 找到当前选中层实际接收到的输入张量，用于对比这一层“进来什么、输出什么”。 */
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

  /** 将三维张量转成图片放入 LLM 上下文，方便解释卷积层或池化层的特征图。 */
  private tensorImageUrlForContext(tensor: ForwardTensor | null): string {
    if (!tensor || tensor.shape.length !== 3) return '';
    return this.tensorToImageDataUrl(tensor, !(tensor.shape[2] === 3 && tensor.colorMode === 'rgb'));
  }

  /** 生成张量摘要，只保留 kind、shape 和前几个数值，避免把完整大张量传给 LLM。 */
  private tensorSummary(tensor: ForwardTensor | null): string {
    if (!tensor) return '无';
    const values = tensor.values ?? [];
    const preview = values.slice(0, 12).map(value => Number(value).toFixed(4)).join(', ');
    return `kind=${tensor.kind}, shape=[${tensor.shape.join(', ')}], values[0..${Math.min(values.length, 12)}]=${preview}${values.length > 12 ? '...' : ''}`;
  }

  /** 读取当前卷积输出通道的 bias；bias 会整体平移该通道特征图响应。 */
  convBiasValue(layer: Extract<NetworkLayer, { type: 'conv2d' }>): number {
    return layer.params.bias?.[this.selectedKernelOutChannel] ?? 0;
  }

  /** 修改卷积 bias 后重新 forward，让页面立即显示偏置对特征图响应的影响。 */
  onConvBiasInput(layer: Extract<NetworkLayer, { type: 'conv2d' }>, value: string | number): void {
    const outChannels = Math.max(1, layer.params.outChannels);
    const bias = layer.params.bias ?? Array.from({ length: outChannels }, () => 0);
    while (bias.length < outChannels) bias.push(0);
    bias[this.selectedKernelOutChannel] = this.finiteNumber(value, bias[this.selectedKernelOutChannel] ?? 0);
    layer.params.bias = bias.slice(0, outChannels);
    this.runForward();
  }

  /** 返回 Dense/Output 神经元索引；每个索引对应权重矩阵中的一行和一个输出单元。 */
  denseUnitIndices(layer: NetworkLayer): number[] {
    if (layer.type !== 'dense' && layer.type !== 'output') return [];
    return Array.from({ length: Math.max(1, layer.params.units) }, (_, index) => index);
  }

  /** 记录当前查看的 Dense 权重行，用来解释某个神经元如何组合全部输入特征。 */
  selectedDenseWeightRow(layer: NetworkLayer): number {
    if (layer.type !== 'dense' && layer.type !== 'output') return 0;
    const max = Math.max(1, layer.params.units) - 1;
    const current = this.selectedDenseWeightRows[layer.id] ?? 0;
    const next = Math.max(0, Math.min(max, current));
    this.selectedDenseWeightRows[layer.id] = next;
    return next;
  }

  /** 切换 Dense 权重行，便于比较不同输出神经元关注的输入特征组合。 */
  onDenseWeightRowChange(layer: NetworkLayer, row: number): void {
    if (layer.type !== 'dense' && layer.type !== 'output') return;
    const max = Math.max(1, layer.params.units) - 1;
    this.selectedDenseWeightRows[layer.id] = Math.max(0, Math.min(max, Number(row) || 0));
  }

  /** 读取 Dense/Output 某个神经元的 bias，它会改变该神经元的激活基线。 */
  denseBiasValue(layer: NetworkLayer, row: number): number {
    if (layer.type !== 'dense' && layer.type !== 'output') return 0;
    return layer.params.bias?.[row] ?? 0;
  }

  /** 修改 Dense/Output bias 并重新 forward，让隐藏响应或类别分数同步刷新。 */
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

  /** 把参数输入安全转换成有限数字，防止 NaN/Infinity 进入卷积或 Dense 计算。 */
  private finiteNumber(value: string | number, fallback: number): number {
    const next = typeof value === 'number' ? value : Number(value);
    return Number.isFinite(next) ? next : fallback;
  }

  /** 按层类型生成教学公式，解释卷积、池化、Flatten、Dense、激活和 Dropout 在 forward 中做什么。 */
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

  /** 从层参数中安全读取数字值，公式面板需要用它展示 kernel、stride、padding、units 等参数。 */
  private numParam(params: Record<string, any> | undefined, key: string, fallback: number): number {
    const value = Number(params?.[key]);
    return Number.isFinite(value) ? value : fallback;
  }

  /** 安全序列化层参数，用于 LLM 上下文和调试信息，避免过长内容撑爆上下文。 */
  private safeJson(value: unknown): string {
    try {
      const text = JSON.stringify(value);
      return text.length > 900 ? `${text.slice(0, 900)}...` : text;
    } catch {
      return String(value);
    }
  }

  /** 判断当前输入是否是 RGB 图像；RGB 输入意味着第一层卷积要处理三个颜色通道。 */
  get isRgbInput(): boolean { return (this.currentInputAsset?.originalChannels ?? 1) >= 3; }
  /** 返回后端校验问题，例如卷积输出尺寸非法、残差分支 shape 不一致等。 */
  get validationIssues(): LayerValidationIssue[] { return this.forwardResult?.validationIssues ?? []; }

  /** 将校验问题按 layerId/field 建索引，右侧参数面板可据此高亮具体输入框。 */
  get fieldIssueMap(): Record<number, Record<string, string[]>> {
    const map: Record<number, Record<string, string[]>> = {};
    for (const issue of this.validationIssues) {
      if (!issue.field) continue;
      map[issue.layerId] ??= {};
      map[issue.layerId][issue.field] = [...(map[issue.layerId][issue.field] ?? []), issue.message];
    }
    return map;
  }

  /** 汇总存在错误的层 id，让网络图和层列表能直接标出 forward 阻塞点。 */
  get errorLayerIdList(): number[] {
    const ids = new Set(this.validationIssues.filter(i => i.severity === 'error').map(i => i.layerId));
    for (const err of this.forwardResult?.errors ?? []) {
      const layerName = err.split(':')[0]?.trim();
      const layer = this.layers.find(l => l.name === layerName);
      if (layer) ids.add(layer.id);
    }
    return [...ids];
  }

  /** 按层收集错误消息，用于解释为什么某一层无法继续完成前向传播。 */
  get layerErrors(): Record<number, string[]> {
    const map: Record<number, string[]> = {};
    for (const issue of this.validationIssues.filter(i => i.severity === 'error')) {
      map[issue.layerId] = [...(map[issue.layerId] ?? []), issue.message];
    }
    return map;
  }

  /** 判断某层是否有校验错误，供 UI 标红问题层。 */
  hasLayerError(id: number): boolean { return !!(this.layerErrors[id]?.length); }
  /** 判断某个层参数是否有错误，例如 kernelSize、stride、padding 或 dropout rate。 */
  hasFieldError(layerId: number, field: string): boolean { return !!(this.fieldIssueMap[layerId]?.[field]?.length); }
  /** 返回字段错误文案，让用户知道当前参数为什么不符合 forward 计算要求。 */
  fieldErrorText(layerId: number, field: string): string { return this.fieldIssueMap[layerId]?.[field]?.[0] ?? ''; }
  /** 返回全局 forward 错误，例如图结构或后端执行过程中无法归属到单个字段的问题。 */
  get globalErrorMessages(): string[] { return this.forwardResult?.errors ?? []; }

  /** 根据当前认证模式显示登录或注册标题；保存历史记录需要绑定用户。 */
  get authTitle(): string { return this.authMode === 'login' ? '登录' : '注册'; }
  /** 只有已有完整 forward 结果且没有待计算改动时，才允许保存实验快照。 */
  get canSaveForwardRecord(): boolean {
    return this.mode === 'forward' && !!this.forwardResult && !this.forwardBusy && !this.pendingForwardChanges;
  }

  /** 打开登录/注册弹窗，用户登录后才能把 A 模式实验记录保存到后端。 */
  openAuthModal(mode: 'login' | 'register' = 'login'): void {
    this.authMode = mode;
    this.authError = '';
    this.authDraft = { username: '', password: '', displayName: '' };
    this.showAuthModal = true;
  }

  /** 关闭认证弹窗；认证请求进行中时不允许关闭，避免状态丢失。 */
  closeAuthModal(): void {
    if (this.authBusy) return;
    this.showAuthModal = false;
    this.authError = '';
  }

  /** 提交登录或注册请求，成功后页面会带 JWT 访问个人历史记录。 */
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

  /** 退出登录并清空本地历史记录列表，避免继续显示上一个用户保存的 A 模式快照。 */
  logout(): void {
    this.authSvc.logout();
    this.forwardRecords = [];
    this.showRecordDrawer = false;
    this.recordSuccess = '已退出登录';
  }

  /** 打开保存记录弹窗；保存的是完整实验快照，包括网络结构、输入图和 forward 结果。 */
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

  /** 关闭保存记录弹窗，保存请求进行中时不关闭以免打断快照提交。 */
  closeSaveRecordModal(): void {
    if (this.recordBusy) return;
    this.showSaveRecordModal = false;
    this.recordError = '';
  }

  /** 保存当前 A 模式实验快照到登录用户的历史记录。 */
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

  /** 打开/关闭历史记录抽屉；记录按用户隔离，只有登录后才能读取自己的实验快照。 */
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

  /** 加载当前登录用户保存过的 A 模式历史记录。 */
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

  /** 从历史记录详情恢复 A 模式页面快照。 */
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

  /** 删除当前用户的一条 A 模式快照记录，不影响当前页面正在编辑的网络。 */
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

  /** 将后端保存的预览图路径转成可访问 URL，用于历史记录列表缩略图。 */
  recordImageUrl(record: ForwardRecordSummary): string {
    return this.forwardRecordSvc.imageUrl(record.imagePath);
  }

  /** 展开当前层全部通道；卷积层的每个通道通常对应一个卷积核提取到的特征图。 */
  openSelectedChannelsModal(): void {
    const tensor = this.selectedForwardResult?.tensor;
    if (!tensor || tensor.shape.length !== 3) return;
    this.channelModalTitle = `${this.selectedForwardResult?.layerName ?? '当前层'} · 全部通道 (${tensor.shape[2]})`;
    this.channelModalPreviews = this.buildChannelPreviews(tensor);
    this.showChannelModal = true;
  }

  /** 展开最终输出的全部通道，便于查看网络最后仍保留空间特征时的每个响应面。 */
  openFinalChannelsModal(): void {
    const tensor = this.forwardResult?.finalTensor;
    if (!tensor || tensor.shape.length !== 3) return;
    this.channelModalTitle = `最终输出 · 全部通道 (${tensor.shape[2]})`;
    this.channelModalPreviews = this.buildChannelPreviews(tensor);
    this.showChannelModal = true;
  }

  /** 关闭通道弹窗并清理通道预览，避免旧特征图残留。 */
  closeChannelModal(): void {
    this.showChannelModal = false;
    this.channelModalPreviews = [];
    this.channelModalTitle = '';
  }

  /** 打开卷积核对比弹窗，针对当前卷积层比较不同滤波器的特征响应。 */
  async openKernelCompareModal(): Promise<void> {
    if (!this.selectedConvLayer) return;
    this.showKernelCompareModal = true;
    await this.runKernelCompare();
  }

  /** 关闭卷积核对比并递增请求序号，防止旧的异步 forward 结果回填到已关闭弹窗。 */
  closeKernelCompareModal(): void {
    this.kernelCompareRequestSeq += 1;
    this.kernelCompareBusy = false;
    this.showKernelCompareModal = false;
    this.kernelCompareError = '';
  }

  /** 克隆当前网络并临时替换卷积核，分别执行 forward，对比边缘、锐化、模糊等 kernel 的输出差异。 */
  async runKernelCompare(): Promise<void> {
    const layer = this.selectedConvLayer;
    const inputTensor = this.currentInputAsset?.prepared.tensor;
    if (!layer || !inputTensor) {
      this.kernelCompareError = '请先选择一个卷积层并准备输入图片。';
      return;
    }

    const requestSeq = ++this.kernelCompareRequestSeq;
    this.kernelCompareBusy = true;
    this.kernelCompareError = '';
    this.kernelCompareItems = [];
    const presets = this.kernelPresets.filter(preset =>
      ['Identity', 'Edge Detect', 'Sharpen', 'Box Blur', 'Sobel X', 'Sobel Y'].includes(preset.label)
    );

    try {
      const items = await Promise.all(presets.map(async preset => {
        const layers = this.layersForKernelCompare(layer.id, preset.matrix);
        const result = await this.forwardBackend.executeForward({
          layers,
          connections: structuredClone(this.connections),
          inputTensor
        });
        const layerResult = result.layerResults.find(item => item.layerId === layer.id);
        if (!layerResult || layerResult.tensor.shape.length !== 3) {
          throw new Error(`${preset.label} 未返回可视化卷积输出。`);
        }
        const [h, w, c] = layerResult.tensor.shape as [number, number, number];
        const channel = Math.min(this.selectedKernelOutChannel, c - 1);
        const values = this.normalizeChannel(this.extractChannel(layerResult.tensor.values, h, w, c, channel));
        return {
          label: preset.label,
          matrix: preset.matrix,
          imageUrl: this.grayValuesToImageDataUrl(values, w, h),
          outputShapeLabel: layerResult.outputShapeLabel,
          stats: layerResult.stats
        };
      }));
      if (requestSeq !== this.kernelCompareRequestSeq || !this.showKernelCompareModal) return;
      this.kernelCompareItems = items;
    } catch (err) {
      if (requestSeq !== this.kernelCompareRequestSeq || !this.showKernelCompareModal) return;
      this.kernelCompareError = err instanceof Error ? err.message : '卷积核对比计算失败。';
    } finally {
      if (requestSeq === this.kernelCompareRequestSeq) {
        this.kernelCompareBusy = false;
      }
    }
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

  /** 删除当前隐藏层并重建拓扑；输入层和输出层是 forward 图的边界，不能删除。 */
  removeSelectedLayer(): void {
    const t = this.selectedLayer;
    if (!t || t.type === 'input' || t.type === 'output') return;
    this.layers = this.layers.filter(l => l.id !== t.id);
    this.selectedLayerId = this.layers[1]?.id ?? this.layers[0]?.id ?? -1;
    this.rebuildTopology(); this.runForward();
  }

  /** 调整当前层在顺序网络中的位置，位置变化会改变张量经过各层的先后顺序。 */
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

  /** 选中某一层，让右侧检查器显示该层参数、公式、shape 和特征图。 */
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

  /** 层参数变化后同步 kernel/input shape，并重新安排 forward，让结果与当前配置一致。 */
  onLayerConfigChange(): void {
    this.syncConvKernelSelectors();
    this.syncKernelShape();
    this.rebuildInputAsset();
    this.runForward();
  }

  /** 修改卷积核尺寸后重建 kernel 矩阵；kernelSize 会直接影响感受野和输出 shape。 */
  onKernelSizeChange(): void { this.syncKernelShape(); this.runForward(); }

  /** 修改卷积核单元格权重；这些数字决定局部窗口内像素如何被加权求和。 */
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

  /** 切换正在编辑的输入/输出通道 kernel，多通道卷积需要分别查看每组权重。 */
  onKernelChannelChange(): void {
    this.syncConvKernelSelectors();
    const l = this.selectedConvLayer;
    if (!l) return;
    this.ensureConvKernelBank(l);
  }

  /** 应用预设卷积核，例如边缘检测、模糊或锐化，用来直观看 kernel 权重对特征图的影响。 */
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

  /** 选择内置样本并重建输入张量，让同一网络对新的图片执行前向传播。 */
  chooseSample(id: number): void {
    this.selectedSampleId = id;
    this.uploadedImageUrl = ''; this.uploadedImageData = null; this.uploadError = '';
    this.clearLocalImageSelection();
    this.showSamplePicker = false;
    this.rebuildInputAsset(); this.runForward();
  }

  /** 打开或收起样本选择器，样本变化会影响输入张量和后续特征图。 */
  toggleSamplePicker(): void { this.showSamplePicker = !this.showSamplePicker; }
  /** 关闭样本选择器，不改变当前 forward 输入。 */
  closeSamplePicker(): void  { this.showSamplePicker = false; }

  /** 读取用户上传图片，解码为 ImageData 后转成网络输入张量，再触发一次 forward。 */
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

  /** 选择 public 目录中的本地图像样本，解码后作为 CNN 输入观察特征提取过程。 */
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

  /** 手动触发真实 forward 请求，把当前 layers、connections 和输入张量提交给后端计算。 */
  triggerForwardCompute(): void {
    this.runForward(true);
  }

  /** 取消本轮 forward 回填；通过递增请求序号让稍后返回的旧结果失效。 */
  cancelForwardCompute(): void {
    this.forwardRerunRequested = false;
    this.forwardBusy = false;
    this.forwardStatusMessage = '计算已取消。';
    this.forwardRequestSeq += 1;
  }

  /** 开启自动计算时，如果已有未计算的参数改动，立即补跑一次 forward。 */
  onAutoForwardComputeToggle(): void {
    if (this.autoForwardCompute && this.pendingForwardChanges) {
      this.runForward(true);
    }
  }

  /** 把后端返回的前向传播结果同步到页面展示状态。 */
  private applyForwardResult(result: ForwardPassResult, seq: number): void {
    if (seq !== this.forwardRequestSeq) return;
    this.forwardResult = result;
    this.forwardLayerShapeMap = result.layerShapeMap;
    const hasSelected = result.layerResults.some(r => r.layerId === this.selectedLayerId);
    if (!hasSelected && result.layerResults.length) {
      this.selectedLayerId = result.layerResults[0].layerId;
    }
  }

  /** 构造历史记录快照；不仅保存网络名，还保存 layers、connections、选中层和 forward 结果。 */
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

  /** 从历史快照恢复 A 模式页面状态，使网络结构、输入图、选中层和结果面板回到保存时刻。 */
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

  /** 读取历史记录保存的预览图，并重新解码为输入张量，保证恢复后还能继续 forward。 */
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

  /** 将当前输入张量渲染成 PNG Data URL，作为历史记录缩略图保存到后端。 */
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
  /** 把 0-1 的激活值映射成颜色，用于小型张量格子预览。 */
  cellColor(v: number): string { return SimEngine.cellColor(v); }
  /** 将一组神经元响应归一化到 0-1，方便画成可比较的条形图。 */
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

  /** 切换上传图计算档位后重建输入尺寸；档位越高，送入网络的图像张量越大。 */
  onUploadComputeProfileChange(): void {
    const imageData = this.uploadedImageData ?? this.localImageData;
    if (!imageData) return;
    this.applyUploadComputeProfile(imageData.width, imageData.height);
    this.rebuildInputAsset();
    this.runForward();
  }

  /** 重建线性拓扑，确保每层 inputs 指向前一层，形成清晰的顺序前向传播链。 */
  private rebuildTopology(): void {
    this.layers = this.layers.map((l, i) => ({ ...l, inputs: i === 0 ? [] : [this.layers[i - 1].id] }));
    this.connections = SimEngine.rebuildLinearConnections(this.layers);
    this.syncKernelShape();
  }

  /** 为 3D 查看器准备每层 shape；优先使用真实后端结果，没有结果时用前端推导值兜底。 */
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

  /** 为 3D 查看器准备每层快照，包括特征图预览、通道图、统计值和 Top-K。 */
  private buildNetwork3dLayerSnapshots(): Record<number, Network3dLayerSnapshot> {
    const snapshots: Record<number, Network3dLayerSnapshot> = {};
    for (const result of this.forwardResult?.layerResults ?? []) {
      const tensor = result.tensor;
      const isImageTensor = tensor.shape.length === 3;
      const previewImageUrl = isImageTensor
        ? this.tensorToImageDataUrl(tensor, !(tensor.shape[2] === 3 && tensor.colorMode === 'rgb'))
        : undefined;
      snapshots[result.layerId] = {
        layerId: result.layerId,
        inputShapeLabel: result.inputShapeLabel,
        outputShapeLabel: result.outputShapeLabel,
        transitionNote: result.transitionNote,
        paramsSummary: structuredClone(result.paramsSummary),
        warnings: structuredClone(result.warnings),
        stats: structuredClone(result.stats),
        visualizationMode: result.visualization.mode,
        previewImageUrl,
        channelPreviews: (result.visualization.channelPreviews ?? []).slice(0, 8).map(channel => ({
          channel: channel.channel,
          width: channel.width,
          height: channel.height,
          imageUrl: this.grayValuesToImageDataUrl(channel.values, channel.width, channel.height)
        })),
        topK: structuredClone(result.stats.topK.slice(0, 5))
      };
    }
    return snapshots;
  }

  /** 根据 kernelSize/outChannels/inChannels 同步卷积核数组形状，保证多通道卷积权重完整。 */
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

  /** 为卷积核对比克隆当前网络并替换目标层 kernel，避免实验对比污染主网络参数。 */
  private layersForKernelCompare(layerId: number, matrix: number[][]): NetworkLayer[] {
    const layers = structuredClone(this.layers);
    const layer = layers.find(item => item.id === layerId);
    if (!layer || layer.type !== 'conv2d') return layers;
    const k = matrix.length || 3;
    const outChannels = Math.max(1, layer.params.outChannels);
    const inChannels = Math.max(1, this.selectedConvInChannels);
    const selectedOut = Math.min(this.selectedKernelOutChannel, outChannels - 1);
    const current = layer.params.kernels ?? [];
    layer.params.kernelSize = k;
    layer.params.kernels = Array.from({ length: outChannels }, (_, oc) => {
      const srcWeights = current[oc]?.weights ?? [];
      const weights = Array.from({ length: inChannels }, (_, ic) => {
        if (oc === selectedOut) {
          return matrix.map(row => [...row]);
        }
        const src = srcWeights[ic] ?? srcWeights[0] ?? layer.params.kernelMatrix ?? matrix;
        return Array.from({ length: k }, (_, y) => Array.from({ length: k }, (_, x) => src[y]?.[x] ?? 0));
      });
      return { ...current[oc], weights };
    });
    layer.params.kernelMatrix = layer.params.kernels[0].weights[0].map(row => [...row]);
    return layers;
  }

  /** 根据上传图、本地样本或内置样本重建输入资产，最终得到 Python forward 使用的 prepared tensor。 */
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

  /** 读取本地图像 manifest，并默认选择一个样本作为 CNN 前向传播输入。 */
  private async loadLocalImageSamples(): Promise<void> {
    try {
      const response = await fetch('mode-a/samples/manifest.json');
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

  /** 清空本地图像选择状态，回到内置样本或上传图片作为输入。 */
  private clearLocalImageSelection(): void {
    this.selectedLocalImageId = '';
    this.localImageData = null;
    this.localImagePreviewUrl = '';
    this.localImageError = '';
  }

  /** 将 prepared tensor 的尺寸同步回输入层，后续 shape 推导和 3D 展示才能与真实输入一致。 */
  private syncInputShape(): void {
    const il = this.inputLayer, t = this.currentInputAsset?.prepared.tensor;
    if (!il || !t || t.shape.length !== 3) return;
    il.params.height = t.shape[0]; il.params.width = t.shape[1]; il.params.channels = t.shape[2];
    il.params.colorMode = t.shape[2] === 1 ? 'grayscale' : 'rgb';
  }

  /** 从 [H, W, C] 特征图中抽取一个通道，用来显示单个卷积核的响应图。 */
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

  /** 将单通道特征响应归一化到 0-1，便于映射成灰度图。 */
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

  /** 对过大的三维张量做预览下采样，降低 DOM/Canvas 渲染成本，不改变真实计算张量。 */
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

  /** 为小网格预览准备 RGB 颜色或灰度值，用于快速查看输入图和小型特征图。 */
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

  /** 用 Canvas 将三维张量转成图片 URL，避免用大量 DOM 小格子渲染高分辨率特征图。 */
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

  /** 将单通道灰度值数组转成 PNG Data URL，用于通道预览和卷积核对比结果。 */
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

  /** 为三维特征图生成每个通道的灰度预览，通道通常对应不同卷积核提取的特征。 */
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

  /** 根据 fast/balanced/quality/original 档位设置上传图 resize 目标，平衡计算量和细节保留。 */
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

  /** 修正当前卷积核通道选择，避免 outChannels 或输入通道数变化后索引越界。 */
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

  /** 确保卷积层按 [outChannel][inChannel][kernelY][kernelX] 保存权重，支撑多通道卷积编辑。 */
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

  /** 创建新层的默认参数：卷积默认 3x3 kernel，池化默认 2x2，下游 Dense/Output 默认带激活。 */
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
