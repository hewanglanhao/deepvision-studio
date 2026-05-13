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

interface DatasetHistoryOption {
  id: string;
  name: string;
  count: number;
  bestAccuracy: number | null;
  latestCreatedAt: string;
}

@Component({
  selector: 'app-experiment-compare-page',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink, NetworkOverviewComponent, PlatformTopbarComponent],
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
}
