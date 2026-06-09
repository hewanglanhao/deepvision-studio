import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { NetworkOverviewComponent } from '@shared/network/network-overview.component';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import { NETWORK_3D_SESSION_KEY, Network3dPayload } from '@shared/network-3d/network-3d.models';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';
import { TrainingCheckpointSummary, TrainingRuntimeService } from '@shared/training/training-runtime.service';
import { SimEngine } from '@shared/simulation/sim-engine';
import { NetworkLayer, TensorShape } from '@shared/simulation/sim-models';
import { LlmFloatingAssistantComponent, LlmQuickPrompt } from '@shared/llm/llm-floating-assistant.component';
import { LlmChatContext } from '@shared/llm/llm.models';
import { MODE_B_LLM_SYSTEM_PROMPT } from '@shared/llm/llm-prompts';

interface DatasetHistoryOption {
  id: string;
  name: string;
  count: number;
  bestAccuracy: number | null;
  latestCreatedAt: string;
}

type CheckpointMetricKey = 'loss' | 'valLoss' | 'accuracy' | 'valAccuracy' | 'lr' | 'gradientNorm';

@Component({
  selector: 'app-experiment-compare-page',
  standalone: true,
  imports: [
    CommonModule,
    FormsModule,
    RouterLink,
    NetworkOverviewComponent,
    PlatformTopbarComponent,
    LlmFloatingAssistantComponent
  ],
  templateUrl: './experiment-compare-page.component.html',
  styleUrl: './experiment-compare-page.component.css'
})
export class ExperimentComparePageComponent implements OnInit, OnDestroy {
  authUser: AuthUser | null = null;
  checkpoints: TrainingCheckpointSummary[] = [];
  selectedDatasetId = '';
  selectedCheckpointId: number | null = null;
  selectedLayerId: number | null = null;
  loading = false;
  error = '';
  private readonly subs = new Subscription();

  readonly topbarStatusPills = ['Checkpoint 历史', '真实训练记录', '结构对比'];
  readonly experimentLlmSystemPrompt = [
    MODE_B_LLM_SYSTEM_PROMPT,
    '',
    '当前页面是实验对比页。请重点比较不同 checkpoint 的网络结构、超参数、训练状态和指标，识别过拟合、欠拟合、未完成训练与异常结果，并给出有依据的下一轮实验建议。'
  ].join('\n');
  readonly experimentLlmContextProvider = (): LlmChatContext => this.buildExperimentLlmContext();
  readonly experimentLlmQuickPrompts: LlmQuickPrompt[] = [
    { label: '对比实验', question: '请比较当前数据集下的各次训练，指出表现最好的一次及原因。' },
    { label: '分析拟合', question: '请结合训练、验证和测试指标判断这些实验是否存在过拟合或欠拟合。' },
    { label: '调参建议', question: '根据当前实验历史，给出下一轮网络结构和超参数调整建议。' },
    { label: '检查异常', question: '请检查未完成、停止或指标异常的训练记录，并分析可能原因。' }
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
        this.selectedDatasetId = '';
        this.selectedCheckpointId = null;
      }
    }));
    void this.authSvc.restoreSession();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  get datasetOptions(): DatasetHistoryOption[] {
    const groups = new Map<string, DatasetHistoryOption>();
    for (const checkpoint of this.checkpoints) {
      const previous = groups.get(checkpoint.datasetId);
      const bestAccuracy = this.maxAccuracy(previous?.bestAccuracy ?? null, checkpoint.testAccuracy);
      const latestCreatedAt = !previous || new Date(checkpoint.createdAt).getTime() > new Date(previous.latestCreatedAt).getTime()
        ? checkpoint.createdAt
        : previous.latestCreatedAt;
      groups.set(checkpoint.datasetId, {
        id: checkpoint.datasetId,
        name: checkpoint.datasetName,
        count: (previous?.count ?? 0) + 1,
        bestAccuracy,
        latestCreatedAt
      });
    }
    return [...groups.values()].sort((a, b) => new Date(b.latestCreatedAt).getTime() - new Date(a.latestCreatedAt).getTime());
  }

  get selectedDatasetName(): string {
    return this.datasetOptions.find(item => item.id === this.selectedDatasetId)?.name ?? '未选择数据集';
  }

  get selectedDatasetOption(): DatasetHistoryOption | null {
    return this.datasetOptions.find(item => item.id === this.selectedDatasetId) ?? null;
  }

  get selectedDatasetCheckpoints(): TrainingCheckpointSummary[] {
    return this.checkpoints.filter(checkpoint => checkpoint.datasetId === this.selectedDatasetId);
  }

  get selectedCheckpoint(): TrainingCheckpointSummary | null {
    return this.selectedDatasetCheckpoints.find(checkpoint => checkpoint.id === this.selectedCheckpointId) ?? this.selectedDatasetCheckpoints[0] ?? null;
  }

  async loadCheckpoints(): Promise<void> {
    if (!this.authUser) return;
    this.loading = true;
    this.error = '';
    try {
      this.checkpoints = await this.trainingSvc.listCheckpoints();
      if (!this.selectedDatasetId || !this.datasetOptions.some(item => item.id === this.selectedDatasetId)) {
        this.selectedDatasetId = this.datasetOptions[0]?.id ?? '';
      }
      this.ensureSelectedCheckpoint();
    } catch (err) {
      this.error = err instanceof Error ? err.message : '加载实验历史失败。';
    } finally {
      this.loading = false;
    }
  }

  selectDataset(datasetId: string): void {
    this.selectedDatasetId = datasetId;
    this.ensureSelectedCheckpoint();
  }

  selectCheckpoint(checkpointId: number): void {
    if (this.selectedCheckpointId === checkpointId) {
      return;
    }
    this.selectedCheckpointId = checkpointId;
    this.selectedLayerId = null;
  }

  selectNetworkLayer(checkpoint: TrainingCheckpointSummary, layerId: number): void {
    this.selectedCheckpointId = checkpoint.id;
    this.selectedLayerId = layerId;
  }

  logout(): void {
    this.authSvc.logout();
  }

  checkpointLayers(checkpoint: TrainingCheckpointSummary): NetworkLayer[] {
    return checkpoint.layers ?? [];
  }

  checkpointPercent(value: number | null | undefined): string {
    return value === null || value === undefined ? 'N/A' : `${(value * 100).toFixed(1)}%`;
  }

  checkpointBarWidth(value: number | null | undefined): number {
    return Math.max(0, Math.min(100, (value ?? 0) * 100));
  }

  checkpointConfigText(checkpoint: TrainingCheckpointSummary): string {
    const config = checkpoint.config;
    if (!config) return '超参数 N/A';
    return `${config.optimizer ?? 'Optimizer'} · lr=${config.learningRate ?? 'N/A'} · batch=${config.batchSize ?? 'N/A'} · epoch=${config.totalEpochs ?? checkpoint.totalEpochs} · loss=${config.lossFunction ?? 'N/A'}`;
  }

  checkpointSplitText(checkpoint: TrainingCheckpointSummary): string {
    const split = checkpoint.split;
    if (!split) return '划分 N/A';
    return `${Math.round((split.train ?? 0) * 100)}% / ${Math.round((split.val ?? 0) * 100)}% / ${Math.round((split.test ?? 0) * 100)}%`;
  }

  checkpointLayerText(checkpoint: TrainingCheckpointSummary): string {
    return checkpoint.networkDescription || (checkpoint.layerSummary ?? []).join(' -> ') || '暂无结构描述';
  }

  selectedLayerFor(checkpoint: TrainingCheckpointSummary): NetworkLayer | null {
    if (checkpoint.id !== this.selectedCheckpointId || this.selectedLayerId === null) return null;
    return this.checkpointLayers(checkpoint).find(layer => layer.id === this.selectedLayerId) ?? null;
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

  layerParamRows(layer: NetworkLayer): Array<{ label: string; value: string }> {
    const params = layer.params as Record<string, unknown>;
    const rows: Array<{ label: string; value: string }> = [
      { label: '层 ID', value: String(layer.id) },
      { label: '层类型', value: this.layerTypeLabel(layer.type) },
      { label: '输入连接', value: layer.inputs?.length ? layer.inputs.join(', ') : '无' }
    ];
    for (const [key, value] of Object.entries(params)) {
      if (key === 'kernels' || key === 'weights' || key === 'bias' || key === 'preprocessing') {
        rows.push({ label: this.paramLabel(key), value: this.compactValue(value) });
        continue;
      }
      rows.push({ label: this.paramLabel(key), value: this.compactValue(value) });
    }
    return rows;
  }

  openNetwork3dViewer(checkpoint: TrainingCheckpointSummary, event?: MouseEvent): void {
    event?.stopPropagation();
    const layers = structuredClone(this.checkpointLayers(checkpoint));
    if (!layers.length) {
      this.error = '该 checkpoint 没有可展示的网络结构。';
      return;
    }
    const layerShapes = this.buildNetwork3dLayerShapes(layers);
    const payload: Network3dPayload = {
      title: `${checkpoint.datasetName} · 实验网络 3D 展示`,
      sourceMode: 'Experiment Compare',
      createdAt: checkpoint.createdAt,
      inputImageUrl: '',
      inputLabel: checkpoint.name,
      datasetName: checkpoint.datasetName,
      parameterCount: SimEngine.parameterCount(layers),
      layers,
      shapeHints: this.buildNetwork3dShapeHints(layerShapes),
      layerShapes,
      layerSnapshots: {},
      shapePath: this.buildNetwork3dShapePath(layers, layerShapes),
      finalTopK: [],
      selectedLayerId: this.selectedLayerId ?? layers[0]?.id ?? -1
    };

    localStorage.setItem(NETWORK_3D_SESSION_KEY, JSON.stringify(payload));
    window.open('/network-3d', '_blank', 'noopener,noreferrer');
  }

  metricTone(value: number | null | undefined): string {
    if (value === null || value === undefined) return 'metric-empty';
    if (value >= 0.85) return 'metric-good';
    if (value >= 0.6) return 'metric-mid';
    return 'metric-low';
  }

  checkpointStatusClass(checkpoint: TrainingCheckpointSummary): string {
    if (checkpoint.status === 'completed' && checkpoint.epoch >= checkpoint.totalEpochs) return 'status-completed';
    if (checkpoint.status === 'stopped') return 'status-stopped';
    if (checkpoint.epoch < checkpoint.totalEpochs) return 'status-incomplete';
    return 'status-unknown';
  }

  checkpointStatusText(checkpoint: TrainingCheckpointSummary): string {
    if (checkpoint.status === 'completed' && checkpoint.epoch >= checkpoint.totalEpochs) return '已完成';
    if (checkpoint.status === 'stopped') return '异常/停止';
    if (checkpoint.epoch < checkpoint.totalEpochs) return `未完成 ${checkpoint.epoch}/${checkpoint.totalEpochs}`;
    return checkpoint.status || '状态未知';
  }

  checkpointStatusNote(checkpoint: TrainingCheckpointSummary): string {
    const hasHistory = this.checkpointMetricHistory(checkpoint).length > 0;
    if (!hasHistory) return '该历史记录没有保存曲线数据，可能是旧版本 checkpoint 或保存时训练流未写入。';
    if (checkpoint.status === 'stopped') return '该次训练异常结束或被停止，曲线仅展示停止前已经记录的部分。';
    if (checkpoint.epoch < checkpoint.totalEpochs) return '该次训练没有跑满预设 epoch，曲线仅展示已经完成的训练部分。';
    return '';
  }

  checkpointMetricHistory(checkpoint: TrainingCheckpointSummary) {
    return (checkpoint.metricHistory ?? []).filter(point => Number.isFinite(point.step));
  }

  checkpointHasMetricHistory(checkpoint: TrainingCheckpointSummary): boolean {
    return this.checkpointMetricHistory(checkpoint).length > 0;
  }

  checkpointFirstStep(checkpoint: TrainingCheckpointSummary): string {
    const history = this.checkpointMetricHistory(checkpoint);
    return `step ${history[0]?.step ?? 0}`;
  }

  checkpointLastStep(checkpoint: TrainingCheckpointSummary): string {
    const history = this.checkpointMetricHistory(checkpoint);
    return `step ${history[history.length - 1]?.step ?? checkpoint.epoch}`;
  }

  checkpointMiniPolyline(checkpoint: TrainingCheckpointSummary, metric: CheckpointMetricKey, group: CheckpointMetricKey[]): string {
    const history = this.checkpointMetricHistory(checkpoint);
    if (!history.length) return '';
    const [minValue, maxValue] = this.checkpointChartDomain(checkpoint, group);
    const span = Math.max(0.000001, maxValue - minValue);
    const usable = history
      .map((point, index) => ({ point, index, value: Number(point[metric]) }))
      .filter(item => Number.isFinite(item.value));
    if (!usable.length) return '';
    const maxIndex = Math.max(1, history.length - 1);
    if (usable.length === 1) {
      const y = 44 - ((usable[0].value - minValue) / span) * 38;
      return `4,${this.clampChartY(y).toFixed(2)} 96,${this.clampChartY(y).toFixed(2)}`;
    }
    return usable.map(item => {
      const x = 4 + (item.index / maxIndex) * 92;
      const y = 44 - ((item.value - minValue) / span) * 38;
      return `${x.toFixed(2)},${this.clampChartY(y).toFixed(2)}`;
    }).join(' ');
  }

  checkpointChartMaxLabel(checkpoint: TrainingCheckpointSummary, group: CheckpointMetricKey[], digits = 3): string {
    const [, maxValue] = this.checkpointChartDomain(checkpoint, group);
    return maxValue.toFixed(digits);
  }

  private checkpointChartDomain(checkpoint: TrainingCheckpointSummary, metrics: CheckpointMetricKey[]): [number, number] {
    const values = this.checkpointMetricHistory(checkpoint).flatMap(point =>
      metrics.map(metric => Number(point[metric])).filter(Number.isFinite)
    );
    if (!values.length) return [0, 1];
    const min = Math.min(0, ...values);
    const max = Math.max(0.000001, ...values);
    return [min, max * 1.08];
  }

  private clampChartY(value: number): number {
    return Math.max(4, Math.min(46, value));
  }

  private ensureSelectedCheckpoint(): void {
    const rows = this.selectedDatasetCheckpoints;
    if (!rows.length) {
      this.selectedCheckpointId = null;
      this.selectedLayerId = null;
      return;
    }
    if (!this.selectedCheckpointId || !rows.some(row => row.id === this.selectedCheckpointId)) {
      this.selectedCheckpointId = rows[0].id;
      this.selectedLayerId = null;
    }
  }

  private maxAccuracy(left: number | null, right: number | null): number | null {
    if (left === null) return right;
    if (right === null) return left;
    return Math.max(left, right);
  }

  private buildNetwork3dLayerShapes(layers: NetworkLayer[]): Record<number, TensorShape> {
    const shapes: Record<number, TensorShape> = {};
    for (const layer of layers) {
      const inputShapes = (layer.inputs ?? [])
        .map(id => shapes[id])
        .filter((shape): shape is TensorShape => shape !== undefined);
      shapes[layer.id] = SimEngine.inferLayerOutputShape(layer, inputShapes);
    }
    return shapes;
  }

  private buildNetwork3dShapeHints(layerShapes: Record<number, TensorShape>): Record<number, string> {
    const hints: Record<number, string> = {};
    for (const [layerId, shape] of Object.entries(layerShapes)) {
      hints[Number(layerId)] = SimEngine.formatShapeLabel(shape);
    }
    return hints;
  }

  private buildNetwork3dShapePath(layers: NetworkLayer[], layerShapes: Record<number, TensorShape>): string[] {
    return layers.map(layer => `${layer.name}: ${SimEngine.formatShapeLabel(layerShapes[layer.id] ?? [])}`);
  }

  private paramLabel(key: string): string {
    const labels: Record<string, string> = {
      inputKind: '输入类型',
      width: '宽度',
      height: '高度',
      channels: '通道数',
      featureCount: '特征数',
      colorMode: '颜色模式',
      preprocessing: '预处理',
      outChannels: '输出通道',
      kernelSize: '卷积核/窗口',
      stride: '步幅',
      padding: '填充',
      dilation: '膨胀',
      activation: '激活函数',
      activationType: '激活函数',
      useProjection: '1x1 投影',
      mode: '池化方式',
      units: '单元/类别数',
      rate: '丢弃率',
      kernels: '卷积核组',
      weights: '权重矩阵',
      bias: '偏置'
    };
    return labels[key] ?? key;
  }

  private compactValue(value: unknown): string {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'boolean') return value ? '是' : '否';
    if (typeof value === 'number') return Number.isInteger(value) ? String(value) : String(Number(value.toFixed(6)));
    if (typeof value === 'string') return value;
    if (Array.isArray(value)) {
      if (!value.length) return '[]';
      return `数组，共 ${value.length} 项`;
    }
    if (typeof value === 'object') {
      return Object.entries(value as Record<string, unknown>)
        .map(([key, item]) => `${this.paramLabel(key)}=${this.compactValue(item)}`)
        .join('，');
    }
    return String(value);
  }

  private buildExperimentLlmContext(): LlmChatContext {
    const rows = this.selectedDatasetCheckpoints.slice(0, 12);
    const lines = [
      '页面: B 模式 / 实验对比',
      `当前用户: ${this.authUser?.username ?? '未登录'}`,
      `当前数据集: ${this.selectedDatasetName} (${this.selectedDatasetId || '未选择'})`,
      `该数据集历史训练数: ${this.selectedDatasetCheckpoints.length}`,
      `当前选中 checkpoint: ${this.selectedCheckpoint?.name ?? '未选择'}`
    ];

    for (const checkpoint of rows) {
      const history = this.checkpointMetricHistory(checkpoint);
      const lastMetric = history[history.length - 1];
      lines.push(
        '',
        `Checkpoint #${checkpoint.id}: ${checkpoint.name}`,
        `状态: ${this.checkpointStatusText(checkpoint)}; epoch=${checkpoint.epoch}/${checkpoint.totalEpochs}; 创建时间=${checkpoint.createdAt}`,
        `指标: train_loss=${this.llmNumber(checkpoint.trainLoss)}, train_accuracy=${this.checkpointPercent(checkpoint.trainAccuracy)}, val_loss=${this.llmNumber(checkpoint.valLoss)}, val_accuracy=${this.checkpointPercent(checkpoint.valAccuracy)}, test_loss=${this.llmNumber(checkpoint.testLoss)}, test_accuracy=${this.checkpointPercent(checkpoint.testAccuracy)}`,
        `超参数: ${this.checkpointConfigText(checkpoint)}`,
        `数据划分: ${this.checkpointSplitText(checkpoint)}`,
        `网络结构: ${this.checkpointLayerText(checkpoint)}`,
        `曲线记录: ${history.length} 个点${lastMetric ? `; 最后 step=${lastMetric.step}, lr=${this.llmNumber(lastMetric.lr)}, gradient_norm=${this.llmNumber(lastMetric.gradientNorm)}` : ''}`
      );
    }

    if (this.selectedCheckpoint && this.selectedLayerId !== null) {
      const layer = this.selectedLayerFor(this.selectedCheckpoint);
      if (layer) {
        lines.push(
          '',
          `当前选中网络层: ${layer.name} (${this.layerTypeLabel(layer.type)})`,
          `层参数: ${this.layerParamRows(layer).map(row => `${row.label}=${row.value}`).join('; ')}`
        );
      }
    }

    if (this.selectedDatasetCheckpoints.length > rows.length) {
      lines.push('', `其余 ${this.selectedDatasetCheckpoints.length - rows.length} 条训练记录未展开。`);
    }
    return { text: lines.join('\n'), images: [] };
  }

  private llmNumber(value: number | null | undefined): string {
    return value === null || value === undefined || !Number.isFinite(value)
      ? 'N/A'
      : Number(value.toFixed(6)).toString();
  }
}
