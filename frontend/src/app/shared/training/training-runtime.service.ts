/**
 * TrainingRuntimeService
 *
 * Supports both the original frontend MOCK runtime and the Spring backend
 * runtime used by mode B.
 */
import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject, Subscription } from 'rxjs';
import { ApiClientService } from '@core/api/api-client.service';
import { AuthService } from '@core/auth/auth.service';
import { SimEngine } from '@shared/simulation/sim-engine';
import {
  MetricPoint,
  NetworkLayer,
  OptimizerType,
  SchedulerType,
  TrainingConfig,
  TrainingRuntimeState
} from '@shared/simulation/sim-models';

export interface TrainingLog {
  time: string;
  level: 'info' | 'warn' | 'error';
  message: string;
}

export interface TrainingPredictionSample {
  index: number;
  trueIndex: number;
  predictedIndex: number;
  trueLabel: string;
  predictedLabel: string;
  confidence: number;
  correct: boolean;
  name?: string;
  imageUrl?: string;
}

export interface TrainingTestResult {
  jobId: string;
  testLoss: number | null;
  testAccuracy: number | null;
  sampleCount: number;
  samples: TrainingPredictionSample[];
}

export interface InferenceSampleItem {
  index: number;
  trueIndex: number;
  trueLabel: string;
  name?: string;
  imageUrl?: string;
  shape: number[];
  featurePreview?: number[];
  featureNames?: string[];
  featureCount?: number;
  rawHeaders?: string[];
  rawValues?: string[];
  rawPreview?: Array<{ name: string; value: string }>;
}

export interface InferenceSampleListResponse {
  type: 'sample_list';
  datasetId: string;
  sampleCount: number;
  samples: InferenceSampleItem[];
}

export interface SingleInferenceActivationPreview {
  mode: 'image' | 'vector';
  width?: number;
  height?: number;
  channels?: number;
  channelPreviews?: Array<{ channel: number; width: number; height: number; values: number[] }>;
  values?: number[];
  valueCount?: number;
}

export interface SingleInferenceActivation {
  order: number;
  layerId: number;
  layerName: string;
  layerType: string;
  shape: number[];
  stats: {
    min: number;
    max: number;
    mean: number;
    nonZeroRatio: number;
  };
  preview: SingleInferenceActivationPreview;
  topValues: Array<{ index: number; value: number; absValue: number }>;
}

export interface SingleInferenceResult {
  type: 'single_inference';
  jobId: string;
  datasetId: string;
  sample: InferenceSampleItem;
  prediction: {
    trueIndex: number;
    trueLabel: string;
    predictedIndex: number;
    predictedLabel: string;
    confidence: number;
    correct: boolean;
    topK: Array<{ index: number; label: string; probability: number }>;
  };
  activations: SingleInferenceActivation[];
}

export interface BackpropLayerStat {
  layerId: number;
  name: string;
  layerType: string;
  trainable: boolean;
  gradNorm: number;
  gradMean: number;
  gradMax: number;
  gradHistogram?: Array<{ from: number; to: number; count: number }>;
  weightNorm: number;
  updateNorm: number;
  status: 'stable' | 'vanishing' | 'exploding' | 'no_grad';
  paramCount: number;
}

export interface BackpropPredictionInsight {
  trueIndex: number;
  predictedIndex: number;
  confidence: number;
  trueProbability: number;
  correct: boolean;
  explanation: string;
}

export interface TrainingBackpropSnapshot {
  type: 'backprop';
  jobId: string;
  epoch: number;
  totalEpochs: number;
  batch: number;
  totalBatches: number;
  phase: string;
  loss: number;
  optimizer: string;
  scheduler: string;
  learningRate: number;
  lr?: number;
  globalGradNorm: number;
  globalUpdateNorm: number;
  gradientStatus: 'stable' | 'vanishing' | 'exploding';
  layers: BackpropLayerStat[];
  prediction: BackpropPredictionInsight;
}

export interface TrainingCheckpointSummary {
  id: number;
  name: string;
  jobId: string;
  datasetId: string;
  datasetName: string;
  modelSignature: string;
  networkDescription: string;
  layerSummary: string[];
  layers: NetworkLayer[];
  config: (TrainingConfig & { lossFunction?: string }) | null;
  split: { train?: number; val?: number; test?: number } | null;
  testResult: unknown;
  metricHistory?: Array<{
    step: number;
    epoch: number;
    loss: number;
    valLoss?: number | null;
    accuracy: number;
    valAccuracy?: number | null;
    lr: number;
    gradientNorm: number;
  }>;
  status?: 'running' | 'paused' | 'stopped' | 'completed' | string;
  epoch: number;
  totalEpochs: number;
  trainLoss: number | null;
  trainAccuracy: number | null;
  valLoss: number | null;
  valAccuracy: number | null;
  testLoss: number | null;
  testAccuracy: number | null;
  testSampleCount: number;
  createdAt: string;
}

export interface BackendTrainingStartRequest {
  datasetId: string;
  split: { train: number; val: number; test: number };
  layers: NetworkLayer[];
  connections: unknown[];
  config: TrainingConfig & { lossFunction?: string };
}

interface BackendTrainingStartResponse {
  jobId: string;
  status: 'running' | 'paused' | 'stopped' | 'completed';
  totalEpochs: number;
  totalBatches: number;
  streamUrl: string;
}

interface BackendMetricMessage {
  type: 'metric' | 'control' | 'error' | 'test_result' | 'backprop';
  jobId: string;
  step?: number;
  epoch?: number;
  batch?: number;
  totalEpochs?: number;
  totalBatches?: number;
  loss?: number;
  valLoss?: number | null;
  accuracy?: number;
  valAccuracy?: number | null;
  lr?: number;
  elapsedSeconds?: number;
  etaSeconds?: number;
  gradientNorm?: number;
  weightMean?: number;
  weightStd?: number;
  gradientStatus?: 'stable' | 'vanishing' | 'exploding';
  status?: 'running' | 'paused' | 'stopped' | 'completed';
  message?: string;
  testLoss?: number | null;
  testAccuracy?: number | null;
  sampleCount?: number;
  samples?: TrainingPredictionSample[];
}

interface BackendControlResponse {
  jobId: string;
  status: 'running' | 'paused' | 'stopped' | 'completed';
  message: string;
}

@Injectable({ providedIn: 'root' })
export class TrainingRuntimeService implements OnDestroy {
  private timer: ReturnType<typeof setInterval> | null = null;
  private socket: WebSocket | null = null;
  private backendJobId = '';
  private backendTotalEpochs = 0;
  private backendTotalBatches = 0;
  private activeUsername = '';
  private readonly privateImageUrls = new Map<string, Promise<string>>();
  private imageUrlToken = '';
  private readonly authSubscription: Subscription;

  readonly state$ = new BehaviorSubject<TrainingRuntimeState>({
    status: 'idle',
    currentEpoch: 0,
    currentLr: 0.001,
    latestLoss: 1.7,
    latestValLoss: 1.78,
    latestAccuracy: 0.2,
    latestValAccuracy: 0.18,
    latestGradientNorm: 1.2,
    latestWeightMean: 0,
    latestWeightStd: 0.16,
    elapsedSeconds: 0,
    etaSeconds: 0,
    message: 'Ready.'
  });

  readonly history$ = new BehaviorSubject<MetricPoint[]>([]);
  readonly logs$ = new BehaviorSubject<TrainingLog[]>([]);
  readonly testResult$ = new BehaviorSubject<TrainingTestResult | null>(null);
  readonly backprop$ = new BehaviorSubject<TrainingBackpropSnapshot | null>(null);
  readonly epochTick$ = new Subject<MetricPoint>();

  private config: TrainingConfig = {
    batchSize: 32,
    totalEpochs: 20,
    learningRate: 0.001,
    optimizer: 'Adam',
    scheduler: 'none',
    lrDecay: 0.9
  };
  private layers: NetworkLayer[] = [];

  constructor(
    private api: ApiClientService,
    private auth: AuthService
  ) {
    this.activeUsername = this.auth.currentUser?.username ?? '';
    this.authSubscription = this.auth.user$.subscribe(user => {
      const username = user?.username ?? '';
      if (username === this.activeUsername) return;
      this.activeUsername = username;
      this.clearClientSession();
    });
  }

  get currentBackendJobId(): string {
    return this.backendJobId;
  }

  // 缓存训练配置和网络层，供后续启动训练或本地状态展示使用。
  configure(config: TrainingConfig, layers: NetworkLayer[]): void {
    this.config = { ...config };
    this.layers = layers;
  }

  // 在开始训练前重置运行时状态、历史曲线、日志和已有 WebSocket 连接。
  prepare(config: TrainingConfig, layers: NetworkLayer[]): void {
    this.clearTimer();
    this.closeSocket();
    this.backendJobId = '';
    this.backendTotalEpochs = 0;
    this.backendTotalBatches = 0;
    this.config = { ...config };
    this.layers = [...layers];
    this.history$.next([]);
    this.logs$.next([]);
    this.testResult$.next(null);
    this.backprop$.next(null);
    this.releaseAllPrivateImageUrls();
    this.patchState({
      status: 'idle',
      currentEpoch: 0,
      currentLr: config.learningRate,
      latestLoss: 1.7,
      latestValLoss: 1.78,
      latestAccuracy: 0.2,
      latestValAccuracy: 0.18,
      latestGradientNorm: 1.2,
      latestWeightMean: 0,
      latestWeightStd: 0.16,
      elapsedSeconds: 0,
      etaSeconds: 0,
      currentBatch: 0,
      totalBatches: 0,
      totalEpochs: config.totalEpochs,
      message: 'Ready.'
    });
  }

  start(): void {
    this.closeSocket();
    this.backendJobId = '';
    const s = this.state$.value;
    if (s.status === 'running') return;
    this.patchState({ status: 'running', message: '[MOCK] Training started.' });
    this.log('info', `[MOCK] Start — optimizer=${this.config.optimizer}, lr=${this.config.learningRate}, epochs=${this.config.totalEpochs}`);
    this.startMock();
  }

  // 向 Spring 后端提交真实训练请求，并在获得 jobId 后连接训练指标 WebSocket。
  async startBackend(request: BackendTrainingStartRequest): Promise<void> {
    this.clearTimer();
    this.closeSocket();
    this.backendJobId = '';
    this.backendTotalEpochs = request.config.totalEpochs;
    this.backendTotalBatches = 0;
    this.config = { ...request.config };
    this.layers = [...request.layers];
    this.history$.next([]);
    this.testResult$.next(null);
    this.backprop$.next(null);
    this.releaseAllPrivateImageUrls();
    this.patchState({
      status: 'running',
      currentEpoch: 0,
      currentLr: request.config.learningRate,
      latestLoss: 1.7,
      latestValLoss: 1.78,
      latestAccuracy: 0.2,
      latestValAccuracy: 0.18,
      latestGradientNorm: 1.2,
      latestWeightMean: 0,
      latestWeightStd: 0.16,
      elapsedSeconds: 0,
      etaSeconds: 0,
      currentBatch: 0,
      totalBatches: 0,
      totalEpochs: request.config.totalEpochs,
      message: 'Starting backend training...'
    });
    this.log('info', `提交训练任务：dataset=${request.datasetId}, optimizer=${request.config.optimizer}, lr=${request.config.learningRate}, epochs=${request.config.totalEpochs}`);

    try {
      const response = await this.api.request<BackendTrainingStartResponse>('/api/training/start', {
        method: 'POST',
        body: JSON.stringify(request)
      });
      this.backendJobId = response.jobId;
      this.backendTotalEpochs = response.totalEpochs;
      this.backendTotalBatches = response.totalBatches;
      this.patchState({
        status: response.status,
        currentBatch: 0,
        totalBatches: response.totalBatches,
        totalEpochs: response.totalEpochs,
        message: `Backend training started: ${response.jobId}`
      });
      this.log('info', `后端训练任务已启动：${response.jobId}`);
      this.connectWebSocket(response.streamUrl);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start backend training.';
      this.patchState({ status: 'stopped', message });
      this.log('error', message);
      throw err;
    }
  }

  // 查询当前用户保存过的 checkpoint，可按数据集过滤用于实验对比。
  async listCheckpoints(datasetId?: string): Promise<TrainingCheckpointSummary[]> {
    const query = datasetId ? `?datasetId=${encodeURIComponent(datasetId)}` : '';
    return this.api.request<TrainingCheckpointSummary[]>(`/api/training/checkpoints${query}`);
  }

  // 触发后端用指定 checkpoint 重新跑测试集，并把预测样本预览规范化。
  async testCheckpoint(checkpointId: number, request: Pick<BackendTrainingStartRequest, 'datasetId' | 'layers'>): Promise<TrainingTestResult> {
    const result = await this.api.request<TrainingTestResult>(`/api/training/checkpoints/${encodeURIComponent(String(checkpointId))}/test`, {
      method: 'POST',
      body: JSON.stringify(request)
    });
    const normalized: TrainingTestResult = {
      ...result,
      samples: await this.normalizePredictionSamples(result.samples ?? [])
    };
    this.testResult$.next(normalized);
    const accText = normalized.testAccuracy === null ? 'N/A' : `${(normalized.testAccuracy * 100).toFixed(1)}%`;
    this.log('info', `已使用 checkpoint 跑测试集：accuracy=${accText}, samples=${normalized.sampleCount}`);
    return normalized;
  }

  // 获取 checkpoint 可推理的样本列表，供单样本推理页面选择。
  async listInferenceSamples(checkpointId: number, limit = 60): Promise<InferenceSampleListResponse> {
    const result = await this.api.request<InferenceSampleListResponse>(
      `/api/training/checkpoints/${encodeURIComponent(String(checkpointId))}/samples?limit=${encodeURIComponent(String(limit))}`
    );
    return {
      ...result,
      samples: await this.normalizeInferenceSamples(result.samples ?? [])
    };
  }

  // 对 checkpoint 中的指定样本执行单样本推理，并返回预测结果与层激活。
  async inferCheckpointSample(checkpointId: number, sampleIndex: number): Promise<SingleInferenceResult> {
    const result = await this.api.request<SingleInferenceResult>(
      `/api/training/checkpoints/${encodeURIComponent(String(checkpointId))}/infer`,
      {
        method: 'POST',
        body: JSON.stringify({ sampleIndex })
      }
    );
    return {
      ...result,
      sample: await this.normalizeInferenceSample(result.sample),
      activations: result.activations ?? []
    };
  }

  // 连接训练任务拥有者的指标流，用于刷新后继续观察已有训练。
  observeBackendJob(jobId: string): void {
    this.beginBackendObservation(jobId, '/api/training/stream');
  }

  // 连接协作房间的旁观指标流，clientId 用于后端确认房间成员身份。
  observeCollaborationJob(jobId: string, clientId: string): void {
    const query = `clientId=${encodeURIComponent(clientId)}`;
    this.beginBackendObservation(jobId, `/api/training/collaboration/stream?${query}`);
  }

  // 断开后端训练观察流，并清空当前观察任务的本地状态。
  disconnectBackendObservation(): void {
    this.clearTimer();
    this.closeSocket();
    this.backendJobId = '';
    this.backendTotalEpochs = 0;
    this.backendTotalBatches = 0;
    this.history$.next([]);
    this.logs$.next([]);
    this.testResult$.next(null);
    this.backprop$.next(null);
    this.releaseAllPrivateImageUrls();
    this.patchState({
      status: 'idle',
      currentEpoch: 0,
      currentBatch: 0,
      totalBatches: 0,
      totalEpochs: 0,
      elapsedSeconds: 0,
      etaSeconds: 0,
      message: 'Training observation disconnected.'
    });
  }

  // 统一初始化已有训练任务的观察状态，并拼接正确的 WebSocket 查询参数。
  private beginBackendObservation(jobId: string, streamPath: string): void {
    const target = jobId.trim();
    if (!target) return;
    this.clearTimer();
    this.closeSocket();
    this.backendJobId = target;
    this.backendTotalEpochs = 0;
    this.backendTotalBatches = 0;
    this.history$.next([]);
    this.logs$.next([]);
    this.testResult$.next(null);
    this.backprop$.next(null);
    this.releaseAllPrivateImageUrls();
    this.patchState({
      status: 'running',
      currentEpoch: 0,
      currentBatch: 0,
      totalBatches: 0,
      totalEpochs: 0,
      message: `Observing backend training: ${target}`
    });
    this.log('info', `加入训练观察：${target}`);
    const separator = streamPath.includes('?') ? '&' : '?';
    this.connectWebSocket(`${streamPath}${separator}jobId=${encodeURIComponent(target)}`);
  }

  // 暂停当前训练；有后端 job 时转发到 Spring 控制接口。
  async pause(): Promise<void> {
    if (this.backendJobId) {
      await this.controlBackend('pause', 'Training paused.');
      return;
    }
    this.clearTimer();
    if (this.state$.value.status === 'running') {
      this.patchState({ status: 'paused', message: 'Paused.' });
      this.log('info', 'Training paused.');
    }
  }

  // 恢复当前训练；后端 job 通过控制接口恢复，本地 mock 则重启定时器。
  async resume(): Promise<void> {
    if (this.backendJobId) {
      await this.controlBackend('resume', 'Training resumed.');
      return;
    }
    if (this.state$.value.status === 'paused') {
      this.patchState({ status: 'running', message: 'Resumed.' });
      this.log('info', 'Training resumed.');
      this.startMock();
    }
  }

  // 停止当前训练，关闭后端指标流或清空本地 mock 状态。
  async stop(): Promise<void> {
    if (this.backendJobId) {
      await this.controlBackend('stop', 'Training stopped.');
      this.closeSocket();
      return;
    }
    this.clearTimer();
    this.patchState({
      status: 'stopped',
      currentEpoch: 0,
      currentLr: this.config.learningRate,
      latestLoss: 1.7,
      latestValLoss: 1.78,
      latestAccuracy: 0.2,
      latestValAccuracy: 0.18,
      latestGradientNorm: 1.2,
      latestWeightMean: 0,
      latestWeightStd: 0.16,
      elapsedSeconds: 0,
      etaSeconds: 0,
      message: 'Stopped.'
    });
    this.history$.next([]);
    this.testResult$.next(null);
    this.backprop$.next(null);
    this.log('warn', 'Training stopped and reset.');
  }

  // 重置训练运行时状态；若后端任务仍在运行，先发送停止命令。
  async reset(): Promise<void> {
    if (this.backendJobId) {
      await this.controlBackend('stop', 'Training stopped.');
    }
    this.clearClientSession();
  }

  ngOnDestroy(): void {
    this.authSubscription.unsubscribe();
    this.clearTimer();
    this.closeSocket();
    this.releaseAllPrivateImageUrls();
  }

  // 清除当前用户相关的训练缓存，登录用户切换时防止串用旧任务状态。
  private clearClientSession(): void {
    this.clearTimer();
    this.closeSocket();
    this.backendJobId = '';
    this.backendTotalEpochs = 0;
    this.backendTotalBatches = 0;
    this.history$.next([]);
    this.logs$.next([]);
    this.testResult$.next(null);
    this.backprop$.next(null);
    this.releaseAllPrivateImageUrls();
    this.patchState({
      status: 'idle',
      currentEpoch: 0,
      currentLr: this.config.learningRate,
      latestLoss: 1.7,
      latestValLoss: 1.78,
      latestAccuracy: 0.2,
      latestValAccuracy: 0.18,
      latestGradientNorm: 1.2,
      latestWeightMean: 0,
      latestWeightStd: 0.16,
      elapsedSeconds: 0,
      etaSeconds: 0,
      currentBatch: 0,
      totalBatches: 0,
      totalEpochs: this.config.totalEpochs,
      message: 'Ready.'
    });
  }

  // 建立训练指标 WebSocket，并在 URL 上附带 JWT 以通过后端鉴权。
  private connectWebSocket(streamUrl: string): void {
    this.closeSocket();
    const wsUrl = this.normalizeWebSocketUrl(streamUrl);
    const token = this.api.token;
    const authenticatedUrl = token
      ? `${wsUrl}${wsUrl.includes('?') ? '&' : '?'}token=${encodeURIComponent(token)}`
      : wsUrl;
    const socket = new WebSocket(authenticatedUrl);
    this.socket = socket;
    socket.onopen = () => this.log('info', `训练指标流已连接：${this.backendJobId}`);
    socket.onmessage = event => this.handleBackendMetric(event.data);
    socket.onerror = () => {
      this.log('error', '训练指标流连接异常。');
    };
    socket.onclose = () => {
      if (this.socket === socket) {
        this.socket = null;
      }
      const status = this.state$.value.status;
      if (status === 'running') {
        this.log('warn', '训练指标流已断开。');
      }
    };
  }

  // 解析后端推送的训练事件，并分发到 metric/backprop/test/control/error 状态流。
  private handleBackendMetric(payload: string): void {
    let message: BackendMetricMessage;
    try {
      message = JSON.parse(payload) as BackendMetricMessage;
    } catch {
      this.log('warn', '收到无法解析的训练指标。');
      return;
    }
    if (message.type === 'error') {
      const text = message.message || 'Python training worker failed.';
      this.patchState({ status: 'stopped', message: text });
      this.log('error', text);
      this.closeSocket();
      return;
    }
    if (message.type === 'control') {
      this.patchState({ status: message.status ?? this.state$.value.status, message: message.message ?? 'Training status changed.' });
      this.log('info', message.message ?? 'Training status changed.');
      if (message.status === 'completed' || message.status === 'stopped') {
        this.closeSocket();
      }
      return;
    }
    if (message.type === 'test_result') {
      const baseResult: Omit<TrainingTestResult, 'samples'> = {
        jobId: message.jobId,
        testLoss: message.testLoss ?? null,
        testAccuracy: message.testAccuracy ?? null,
        sampleCount: message.sampleCount ?? 0
      };
      void this.normalizePredictionSamples(message.samples ?? []).then(samples => {
        const result: TrainingTestResult = { ...baseResult, samples };
        this.testResult$.next(result);
        const accText = result.testAccuracy === null ? 'N/A' : `${(result.testAccuracy * 100).toFixed(1)}%`;
        this.log('info', `测试集评估完成：accuracy=${accText}, samples=${result.sampleCount}`);
      });
      return;
    }
    if (message.type === 'backprop') {
      this.backprop$.next(message as TrainingBackpropSnapshot);
      return;
    }
    if (message.type !== 'metric') return;
    if (message.step === undefined || message.epoch === undefined || message.batch === undefined
      || message.totalEpochs === undefined || message.totalBatches === undefined
      || message.loss === undefined || message.accuracy === undefined || message.lr === undefined
      || message.elapsedSeconds === undefined || message.etaSeconds === undefined
      || message.gradientNorm === undefined || message.weightMean === undefined || message.weightStd === undefined) {
      this.log('warn', '收到不完整的训练指标。');
      return;
    }
    const metric: MetricPoint = {
      step: message.step,
      loss: message.loss,
      valLoss: message.valLoss ?? message.loss,
      accuracy: message.accuracy,
      valAccuracy: message.valAccuracy ?? message.accuracy,
      lr: message.lr,
      gradientNorm: message.gradientNorm,
      weightMean: message.weightMean,
      weightStd: message.weightStd,
      elapsedSeconds: message.elapsedSeconds,
      etaSeconds: message.etaSeconds
    };
    this.patchState({
      status: message.epoch >= message.totalEpochs ? 'completed' : 'running',
      currentEpoch: message.epoch,
      currentLr: message.lr,
      latestLoss: message.loss,
      latestValLoss: metric.valLoss,
      latestAccuracy: message.accuracy,
      latestValAccuracy: metric.valAccuracy,
      latestGradientNorm: message.gradientNorm,
      latestWeightMean: message.weightMean,
      latestWeightStd: message.weightStd,
      elapsedSeconds: message.elapsedSeconds,
      etaSeconds: message.etaSeconds,
      currentBatch: message.batch,
      totalBatches: message.totalBatches,
      totalEpochs: message.totalEpochs,
      message: `Backend epoch ${message.epoch}/${message.totalEpochs}`
    });
    const history = [...this.history$.value, metric].slice(-200);
    this.history$.next(history);
    this.epochTick$.next(metric);
    if (message.epoch % 5 === 0 || message.epoch === message.totalEpochs) {
      this.log('info', `Epoch ${message.epoch}: loss=${message.loss.toFixed(4)}, val_loss=${metric.valLoss.toFixed(4)}, acc=${(message.accuracy * 100).toFixed(1)}%, gradient=${message.gradientStatus ?? 'stable'}`);
    }
    if (message.epoch >= message.totalEpochs) {
      this.log('info', `训练轮次完成，等待测试集评估：${message.jobId}`);
    }
  }

  // 调用后端训练控制接口，将暂停、恢复、停止、重置命令传给 Spring。
  private async controlBackend(action: 'pause' | 'resume' | 'stop' | 'reset', fallback: string): Promise<void> {
    if (!this.backendJobId) return;
    try {
      const response = await this.api.request<BackendControlResponse>(`/api/training/${encodeURIComponent(this.backendJobId)}/${action}`, {
        method: 'POST'
      });
      this.patchState({ status: response.status, message: response.message || fallback });
      this.log(action === 'stop' ? 'warn' : 'info', response.message || fallback);
    } catch (err) {
      const message = err instanceof Error ? err.message : `Failed to ${action} training.`;
      this.log('error', message);
      throw err;
    }
  }

  // 主动关闭当前 WebSocket，并解绑回调避免关闭事件重复修改状态。
  private closeSocket(): void {
    if (this.socket) {
      const socket = this.socket;
      this.socket = null;
      socket.onclose = null;
      socket.close();
    }
  }

  // 将后端返回的相对路径或 HTTP 地址转换成浏览器可连接的 ws/wss 地址。
  private normalizeWebSocketUrl(streamUrl: string): string {
    if (streamUrl.startsWith('ws://') || streamUrl.startsWith('wss://')) {
      return streamUrl;
    }
    if (streamUrl.startsWith('/')) {
      const base = this.api.baseUrl
        ? this.api.baseUrl.replace(/^http/, 'ws')
        : `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`;
      return `${base}${streamUrl}`;
    }
    return streamUrl;
  }

  private async normalizePredictionSamples(samples: TrainingPredictionSample[]): Promise<TrainingPredictionSample[]> {
    return Promise.all(samples.map(async sample => ({
      ...sample,
      imageUrl: await this.resolveImageUrl(sample.imageUrl)
    })));
  }

  private async normalizeInferenceSamples(samples: InferenceSampleItem[]): Promise<InferenceSampleItem[]> {
    return Promise.all(samples.map(sample => this.normalizeInferenceSample(sample)));
  }

  private async normalizeInferenceSample(sample: InferenceSampleItem | undefined): Promise<InferenceSampleItem> {
    if (!sample) return sample as unknown as InferenceSampleItem;
    return {
      ...sample,
      imageUrl: await this.resolveImageUrl(sample.imageUrl)
    };
  }

  private async resolveImageUrl(url: string | undefined): Promise<string | undefined> {
    if (!url) return url;
    const normalized = this.normalizeResourceUrl(url);
    if (!this.isPrivateDatasetFileUrl(normalized)) {
      return normalized;
    }
    this.ensureImageUrlCacheOwner();
    let cached = this.privateImageUrls.get(normalized);
    if (!cached) {
      cached = this.fetchPrivateImageUrl(normalized);
      this.privateImageUrls.set(normalized, cached);
    }
    try {
      return await cached;
    } catch {
      this.privateImageUrls.delete(normalized);
      return normalized;
    }
  }

  private normalizeResourceUrl(url: string): string {
    if (!url || /^https?:\/\//i.test(url) || url.startsWith('data:') || url.startsWith('blob:')) return url;
    return `${this.api.baseUrl}${url.startsWith('/') ? url : '/' + url}`;
  }

  private isPrivateDatasetFileUrl(url: string): boolean {
    try {
      const path = new URL(url, window.location.origin).pathname;
      return path.startsWith('/api/training/datasets/') && path.includes('/files/');
    } catch {
      return url.includes('/api/training/datasets/') && url.includes('/files/');
    }
  }

  private async fetchPrivateImageUrl(url: string): Promise<string> {
    const headers = new Headers();
    if (this.api.token) {
      headers.set('Authorization', `Bearer ${this.api.token}`);
    }
    const response = await fetch(url, { headers });
    if (!response.ok) {
      throw new Error(`Image HTTP ${response.status}`);
    }
    return URL.createObjectURL(await response.blob());
  }

  private ensureImageUrlCacheOwner(): void {
    const token = this.api.token;
    if (token === this.imageUrlToken) return;
    this.releaseAllPrivateImageUrls();
    this.imageUrlToken = token;
  }

  private releaseAllPrivateImageUrls(): void {
    for (const imageUrl of this.privateImageUrls.values()) {
      void imageUrl.then(value => URL.revokeObjectURL(value)).catch(() => undefined);
    }
    this.privateImageUrls.clear();
  }

  private startMock(): void {
    this.timer = setInterval(() => {
      const s = this.state$.value;
      if (s.currentEpoch >= this.config.totalEpochs) {
        this.clearTimer();
        this.patchState({ status: 'idle', message: `[MOCK] Finished ${this.config.totalEpochs} epochs.` });
        this.log('info', `[MOCK] Training complete. Final acc=${(s.latestAccuracy * 100).toFixed(1)}%`);
        return;
      }
      const tick = SimEngine.nextTrainingState({
        state: s,
        totalEpochs: this.config.totalEpochs,
        layers: this.layers,
        optimizer: this.config.optimizer as OptimizerType,
        learningRate: this.config.learningRate,
        scheduler: this.config.scheduler as SchedulerType,
        lrDecay: this.config.lrDecay
      });
      this.patchState({
        ...tick.state,
        message: `[MOCK] Epoch ${tick.state.currentEpoch}/${this.config.totalEpochs}`
      });
      const history = [...this.history$.value, tick.metric].slice(-200);
      this.history$.next(history);
      this.epochTick$.next(tick.metric);
      if (tick.state.currentEpoch % 5 === 0) {
        this.log('info', `Epoch ${tick.state.currentEpoch}: loss=${tick.metric.loss.toFixed(4)}, val_loss=${tick.metric.valLoss.toFixed(4)}, acc=${(tick.metric.accuracy * 100).toFixed(1)}%`);
      }
    }, 280);
  }

  private patchState(patch: Partial<TrainingRuntimeState>): void {
    this.state$.next({ ...this.state$.value, ...patch });
  }

  private clearTimer(): void {
    if (this.timer) { clearInterval(this.timer); this.timer = null; }
  }

  private log(level: TrainingLog['level'], message: string): void {
    const now = new Date();
    const time = `${now.getHours().toString().padStart(2, '0')}:${now.getMinutes().toString().padStart(2, '0')}:${now.getSeconds().toString().padStart(2, '0')}`;
    const logs = [...this.logs$.value, { time, level, message }].slice(-100);
    this.logs$.next(logs);
  }
}
