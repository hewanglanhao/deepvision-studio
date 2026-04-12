import { CommonModule } from '@angular/common';
import { Component, OnDestroy } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { HelpManualComponent } from './components/help-manual.component';
import { NetworkOverviewComponent } from './components/network-overview.component';
import { SimEngine } from './sim-engine';
import {
  Connection,
  DataSample,
  ExperimentResult,
  LayerType,
  MetricPoint,
  ModelTemplate,
  NetworkLayer,
  OptimizerType,
  PresetTask,
  SchedulerType
} from './sim-models';

@Component({
  selector: 'app-root',
  imports: [CommonModule, FormsModule, HelpManualComponent, NetworkOverviewComponent],
  templateUrl: './app.html',
  styleUrl: './app.css'
})
export class App implements OnDestroy {
  title = 'DeepVision Studio';

  nextLayerId = 4;
  draggingLayerType: LayerType | null = null;

  modelTemplates: ModelTemplate[] = SimEngine.templates();
  selectedTemplateId = 'mlp-basic';

  layers: NetworkLayer[] = [
    {
      id: 1,
      type: 'input',
      name: 'Input',
      units: 784,
      kernelSize: 0,
      activation: 'none',
      dropoutRate: 0
    },
    {
      id: 2,
      type: 'dense',
      name: 'Dense 1',
      units: 64,
      kernelSize: 0,
      activation: 'relu',
      dropoutRate: 0
    },
    {
      id: 3,
      type: 'output',
      name: 'Output',
      units: 10,
      kernelSize: 0,
      activation: 'softmax',
      dropoutRate: 0
    }
  ];

  connections: Connection[] = [];
  selectedLayerId = 2;
  connectFromId = 1;
  connectToId = 3;

  datasets: Record<string, DataSample[]> = {
    MNIST: SimEngine.generateDataset(10, 64, 'mnist'),
    'CIFAR-10': SimEngine.generateDataset(10, 100, 'cifar')
  };
  selectedDataset = 'MNIST';
  selectedSampleId = 1;

  status: 'idle' | 'running' | 'paused' | 'stopped' = 'idle';
  batchSize = 32;
  totalEpochs = 30;
  optimizer: OptimizerType = 'Adam';
  learningRate = 0.001;
  scheduler: SchedulerType = 'none';
  lrDecay = 0.9;
  currentEpoch = 0;
  currentLr = 0.001;
  trainingHistory: MetricPoint[] = [];
  trainTimer: ReturnType<typeof setInterval> | null = null;
  latestLoss = 0;
  latestAccuracy = 0;
  latestValAccuracy = 0;

  featureMaps: number[][] = [];
  inferenceActivations: { layerName: string; values: number[] }[] = [];
  gradCamMap: number[] = [];
  confusionMatrix: number[][] = [];

  uploadedImageUrl = '';
  uploadedInputMap: number[] = [];
  uploadedConvMap: number[] = [];
  uploadedPoolMap: number[] = [];
  uploadedInputSize = 28;
  uploadedConvSize = 28;
  uploadedPoolSize = 14;
  convKernelType = 'edge';
  convStride = 1;
  private uploadedMatrix: number[][] = [];

  presetTasks: PresetTask[] = [
    {
      id: 'digits-basic',
      name: '任务 A: 手写数字识别',
      type: 'classification',
      dataset: 'MNIST',
      description: '10 类分类，观察深度与激活函数对精度影响。'
    },
    {
      id: 'fashion-lite',
      name: '任务 B: 服饰分类',
      type: 'classification',
      dataset: 'MNIST',
      description: '使用同构网络测试收敛速度。'
    },
    {
      id: 'objects-mini',
      name: '任务 C: 物体粗分类',
      type: 'classification',
      dataset: 'CIFAR-10',
      description: '引入卷积层，比较池化策略。'
    },
    {
      id: 'regression-toy',
      name: '任务 D: 回归拟合',
      type: 'regression',
      dataset: 'MNIST',
      description: '观察 dropout 与学习率衰减对稳定性的影响。'
    }
  ];
  selectedTaskId = 'digits-basic';
  evaluationLogs: string[] = [];

  experimentResults: ExperimentResult[] = [];
  experimentNote = '';

  aiQuestion = '';
  aiAnswer = '输入训练现象后，系统会给出解释和调参建议。';
  showHelp = false;
  lossLandscape: number[][] = [];
  showNetworkModal = false;
  networkZoom = 1;
  selectedNodeKeys: string[] = [];

  constructor() {
    this.rebuildLinearConnections();
    this.trainingHistory = [{ step: 0, loss: 1.7, accuracy: 0.22, valAccuracy: 0.2, lr: this.learningRate }];
    this.latestLoss = 1.7;
    this.latestAccuracy = 0.22;
    this.latestValAccuracy = 0.2;
    this.currentLr = this.learningRate;
    this.confusionMatrix = SimEngine.buildConfusionMatrix(0);
    this.refreshVisuals();
    this.refreshLandscape();
  }

  ngOnDestroy(): void {
    if (this.trainTimer) {
      clearInterval(this.trainTimer);
      this.trainTimer = null;
    }
  }

  get layerPalette(): LayerType[] {
    return ['dense', 'conv', 'pool', 'dropout'];
  }

  get selectedLayer(): NetworkLayer | undefined {
    return this.layers.find((layer) => layer.id === this.selectedLayerId);
  }

  get datasetSamples(): DataSample[] {
    return this.datasets[this.selectedDataset] ?? [];
  }

  get selectedSample(): DataSample | undefined {
    return this.datasetSamples.find((sample) => sample.id === this.selectedSampleId);
  }

  get selectedTask(): PresetTask | undefined {
    return this.presetTasks.find((task) => task.id === this.selectedTaskId);
  }

  get selectedTemplate(): ModelTemplate | undefined {
    return this.modelTemplates.find((tpl) => tpl.id === this.selectedTemplateId);
  }

  get layerCount(): number {
    return this.layers.length;
  }

  get parameterCount(): number {
    return SimEngine.parameterCount(this.layers);
  }

  get lossPolyline(): string {
    return SimEngine.buildPolyline(this.trainingHistory, 'loss');
  }

  get accPolyline(): string {
    return SimEngine.buildPolyline(this.trainingHistory, 'accuracy');
  }

  get valPolyline(): string {
    return SimEngine.buildPolyline(this.trainingHistory, 'valAccuracy');
  }

  get lossScatter(): Array<{ x: number; y: number; loss: number }> {
    if (this.trainingHistory.length === 0) {
      return [];
    }

    const maxStep = Math.max(1, this.trainingHistory[this.trainingHistory.length - 1].step);
    return this.trainingHistory.map((point) => {
      const x = (point.step / maxStep) * 100;
      const y = 100 - Math.min(100, point.loss * 48);
      return { x, y, loss: point.loss };
    });
  }

  onLayerDragStart(type: LayerType): void {
    this.draggingLayerType = type;
  }

  onCanvasDragOver(event: DragEvent): void {
    event.preventDefault();
  }

  onCanvasDrop(event: DragEvent): void {
    event.preventDefault();
    if (!this.draggingLayerType) {
      return;
    }
    this.addLayer(this.draggingLayerType);
    this.draggingLayerType = null;
  }

  applyTemplate(): void {
    const template = this.selectedTemplate;
    if (!template) {
      return;
    }

    let nextId = 1;
    this.layers = template.layers.map((layer) => ({ ...layer, id: nextId++ }));
    this.nextLayerId = nextId;
    this.selectedLayerId = this.layers[1]?.id ?? this.layers[0]?.id ?? 1;
    this.rebuildLinearConnections();
    this.refreshVisuals();
    this.refreshLandscape();
  }

  addLayer(type: LayerType): void {
    if (type === 'input' || type === 'output') {
      return;
    }

    const newLayer: NetworkLayer = {
      id: this.nextLayerId,
      type,
      name: `${type.toUpperCase()} ${this.nextLayerId}`,
      units: type === 'conv' ? 16 : type === 'pool' ? 32 : type === 'dropout' ? 32 : 64,
      kernelSize: type === 'conv' ? 3 : 0,
      activation: type === 'dropout' || type === 'pool' ? 'none' : 'relu',
      dropoutRate: type === 'dropout' ? 0.25 : 0
    };
    this.nextLayerId += 1;

    const outputIndex = this.layers.findIndex((layer) => layer.type === 'output');
    this.layers.splice(outputIndex, 0, newLayer);
    this.selectedLayerId = newLayer.id;
    this.rebuildLinearConnections();
    this.refreshVisuals();
    this.refreshLandscape();
  }

  removeSelectedLayer(): void {
    const layer = this.selectedLayer;
    if (!layer || layer.type === 'input' || layer.type === 'output') {
      return;
    }

    this.layers = this.layers.filter((item) => item.id !== layer.id);
    this.selectedLayerId = this.layers[1]?.id ?? this.layers[0].id;
    this.rebuildLinearConnections();
    this.refreshVisuals();
    this.refreshLandscape();
  }

  adjustLayerUnits(layerId: number, delta: number): void {
    const layer = this.layers.find((item) => item.id === layerId);
    if (!layer) {
      return;
    }
    if (layer.type === 'input' || layer.type === 'output') {
      return;
    }

    layer.units = Math.max(2, Math.min(512, layer.units + delta));
    this.selectedLayerId = layer.id;
    this.refreshVisuals();
    this.refreshLandscape();
  }

  onNodeSelected(event: { layerId: number; nodeIndex: number; append: boolean }): void {
    const key = `${event.layerId}-${event.nodeIndex}`;
    if (!event.append) {
      this.selectedNodeKeys = [key];
      return;
    }

    if (this.selectedNodeKeys.includes(key)) {
      this.selectedNodeKeys = this.selectedNodeKeys.filter((item) => item !== key);
    } else {
      this.selectedNodeKeys = [...this.selectedNodeKeys, key];
    }
  }

  onNetworkPan(event: Event, scroller: HTMLElement): void {
    const input = event.target as HTMLInputElement;
    const ratio = Number(input.value) / 100;
    const max = Math.max(0, scroller.scrollWidth - scroller.clientWidth);
    scroller.scrollLeft = max * ratio;
  }

  resetNodeSelection(): void {
    this.selectedNodeKeys = [];
  }

  addConnection(): void {
    if (this.connectFromId === this.connectToId) {
      return;
    }

    if (this.connectFromId > this.connectToId) {
      return;
    }

    const exists = this.connections.some(
      (conn) => conn.from === this.connectFromId && conn.to === this.connectToId
    );
    if (!exists) {
      this.connections.push({ from: this.connectFromId, to: this.connectToId });
    }
  }

  removeConnection(conn: Connection): void {
    this.connections = this.connections.filter(
      (item) => !(item.from === conn.from && item.to === conn.to)
    );
  }

  selectDataset(name: string): void {
    this.selectedDataset = name;
    this.selectedSampleId = this.datasetSamples[0]?.id ?? 1;
    this.refreshVisuals();
  }

  startTraining(): void {
    if (this.status === 'running') {
      return;
    }
    this.status = 'running';

    if (this.currentEpoch >= this.totalEpochs) {
      this.currentEpoch = 0;
      this.trainingHistory = [{ step: 0, loss: 1.7, accuracy: 0.22, valAccuracy: 0.2, lr: this.learningRate }];
    }

    this.trainTimer = setInterval(() => {
      if (this.currentEpoch >= this.totalEpochs) {
        this.pauseTraining();
        return;
      }
      this.currentEpoch += 1;

      const point = SimEngine.pushMetricPoint({
        currentEpoch: this.currentEpoch,
        totalEpochs: this.totalEpochs,
        layers: this.layers,
        optimizer: this.optimizer,
        learningRate: this.learningRate,
        scheduler: this.scheduler,
        lrDecay: this.lrDecay
      });

      this.currentLr = point.lr;
      this.latestLoss = point.loss;
      this.latestAccuracy = point.accuracy;
      this.latestValAccuracy = point.valAccuracy;
      this.trainingHistory.push(point);
      this.trainingHistory = this.trainingHistory.slice(-120);
      this.confusionMatrix = SimEngine.buildConfusionMatrix(this.currentEpoch);
      this.refreshVisuals();
      this.refreshLandscape();
    }, 260);
  }

  pauseTraining(): void {
    if (this.trainTimer) {
      clearInterval(this.trainTimer);
      this.trainTimer = null;
    }
    if (this.status === 'running') {
      this.status = 'paused';
    }
  }

  stopTraining(): void {
    if (this.trainTimer) {
      clearInterval(this.trainTimer);
      this.trainTimer = null;
    }
    this.status = 'stopped';
    this.currentEpoch = 0;
    this.currentLr = this.learningRate;
    this.trainingHistory = [{ step: 0, loss: 1.7, accuracy: 0.22, valAccuracy: 0.2, lr: this.learningRate }];
    this.latestLoss = 1.7;
    this.latestAccuracy = 0.22;
    this.latestValAccuracy = 0.2;
    this.confusionMatrix = SimEngine.buildConfusionMatrix(0);
    this.refreshVisuals();
    this.refreshLandscape();
  }

  runInference(): void {
    const sample = this.selectedSample;
    const visuals = SimEngine.refreshVisuals({
      sample,
      selectedDataset: this.selectedDataset,
      currentEpoch: this.currentEpoch,
      layers: this.layers
    });
    this.inferenceActivations = visuals.inferenceActivations;
  }

  runSelectedTask(): void {
    const task = this.selectedTask;
    if (!task) {
      return;
    }

    this.selectDataset(task.dataset);
    const acc = SimEngine.evaluateTask(task, this.layers, this.optimizer, this.totalEpochs);
    const msg = `${task.name} 已评估: 测试指标 ${(acc * 100).toFixed(1)}% | 数据集 ${task.dataset} | 轮数 ${this.totalEpochs}`;
    this.evaluationLogs.unshift(msg);
    this.evaluationLogs = this.evaluationLogs.slice(0, 8);
  }

  runStructureExperiment(mode: 'deeper' | 'activation' | 'optimizer'): void {
    const baseline = this.latestValAccuracy || this.latestAccuracy || 0.22;
    const result = SimEngine.runExperiment(mode, baseline, this.totalEpochs);
    this.experimentResults.unshift(result);
    this.experimentResults = this.experimentResults.slice(0, 6);
    this.experimentNote = `已完成对比实验: ${result.name}，验证精度 ${(result.finalAccuracy * 100).toFixed(1)}%，训练速度系数 ${result.speedScore.toFixed(2)}。`;
  }

  askAiHelper(): void {
    const text = this.aiQuestion.toLowerCase();
    const lines: string[] = [];

    if (text.includes('震荡') || text.includes('不稳定') || text.includes('oscillat')) {
      lines.push('现象解释: 损失震荡通常由学习率偏大或 batch 偏小导致梯度方差过高。');
      lines.push('建议: 学习率先降低到当前的 0.5 倍，并将 batch 提升到 64。');
    }
    if (text.includes('过拟合') || text.includes('overfit')) {
      lines.push('现象解释: 训练集高精度而验证集停滞，属于典型过拟合。');
      lines.push('建议: 增加 Dropout 层或早停，尝试 0.2-0.4 的 dropoutRate。');
    }
    if (text.includes('收敛慢') || text.includes('slow')) {
      lines.push('现象解释: 收敛慢常见于特征提取不足或优化器配置不佳。');
      lines.push('建议: 增加一个卷积层并改用 Adam/AdamW，同时启用学习率衰减。');
    }
    if (lines.length === 0) {
      lines.push('现象解释: 当前描述较泛化，建议补充具体指标(训练/验证损失、精度曲线形态)。');
      lines.push('建议: 先运行结构对比实验，再根据结果定位瓶颈。');
    }

    lines.push(
      `当前配置摘要: 层数 ${this.layerCount}，优化器 ${this.optimizer}，调度器 ${this.scheduler}，学习率 ${this.currentLr.toFixed(5)}，batch ${this.batchSize}。`
    );
    this.aiAnswer = lines.join('\n');
  }

  onImageUpload(event: Event): void {
    const input = event.target as HTMLInputElement;
    const file = input.files?.[0];
    if (!file) {
      return;
    }

    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result;
      if (typeof result !== 'string') {
        return;
      }
      this.uploadedImageUrl = result;
      this.decodeAndPrepareUploadedImage(result);
    };
    reader.readAsDataURL(file);
  }

  simulateUploadedConvolution(): void {
    if (this.uploadedMatrix.length === 0) {
      return;
    }

    const kernel = SimEngine.kernelPresets[this.convKernelType] ?? SimEngine.kernelPresets['edge'];
    const conv = SimEngine.convolve2d(this.uploadedMatrix, kernel, {
      stride: this.convStride,
      padding: 'same',
      relu: true
    });
    const pooled = SimEngine.maxPool2d(conv, 2, 2);

    const normConv = SimEngine.normalizeMatrix(conv);
    const normPool = SimEngine.normalizeMatrix(pooled);

    this.uploadedConvSize = normConv.length;
    this.uploadedPoolSize = normPool.length;
    this.uploadedConvMap = SimEngine.flattenMatrix(normConv);
    this.uploadedPoolMap = SimEngine.flattenMatrix(normPool);
  }

  getLayerTypeLabel(type: LayerType): string {
    return SimEngine.layerTypeLabel(type);
  }

  getCellColor(value: number, colorMode: 'mono' | 'heat' = 'mono'): string {
    return SimEngine.cellColor(value, colorMode);
  }

  private rebuildLinearConnections(): void {
    this.connections = SimEngine.rebuildLinearConnections(this.layers);
    this.connectFromId = this.layers[0].id;
    this.connectToId = this.layers[this.layers.length - 1].id;
  }

  refreshVisuals(): void {
    const visuals = SimEngine.refreshVisuals({
      sample: this.selectedSample,
      selectedDataset: this.selectedDataset,
      currentEpoch: this.currentEpoch,
      layers: this.layers
    });

    this.featureMaps = visuals.featureMaps;
    this.gradCamMap = visuals.gradCamMap;
    if (this.inferenceActivations.length === 0) {
      this.inferenceActivations = visuals.inferenceActivations;
    }
  }

  private decodeAndPrepareUploadedImage(imageUrl: string): void {
    const img = new Image();
    img.onload = () => {
      const size = 28;
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      const ctx = canvas.getContext('2d');
      if (!ctx) {
        return;
      }

      ctx.clearRect(0, 0, size, size);
      ctx.drawImage(img, 0, 0, size, size);
      const imageData = ctx.getImageData(0, 0, size, size);
      const grayMatrix = SimEngine.imageDataToGrayMatrix(imageData, true);

      this.uploadedMatrix = grayMatrix;
      this.uploadedInputSize = grayMatrix.length;
      this.uploadedInputMap = SimEngine.flattenMatrix(grayMatrix);
      this.simulateUploadedConvolution();
    };
    img.src = imageUrl;
  }

  private refreshLandscape(): void {
    this.lossLandscape = SimEngine.buildLossLandscape(
      this.latestLoss,
      this.latestValAccuracy || this.latestAccuracy,
      this.layerCount,
      this.optimizer
    );
  }
}
