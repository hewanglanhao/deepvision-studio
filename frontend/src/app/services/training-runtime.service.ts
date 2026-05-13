/**
 * TrainingRuntimeService
 *
 * Supports both the original frontend MOCK runtime and the Spring backend
 * runtime used by mode B.
 */
import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
import { ApiClientService } from './api-client.service';
import { SimEngine } from '../sim-engine';
import {
  MetricPoint,
  NetworkLayer,
  OptimizerType,
  SchedulerType,
  TrainingConfig,
  TrainingRuntimeState
} from '../sim-models';

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
  type: 'metric' | 'control' | 'error' | 'test_result';
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

  constructor(private api: ApiClientService) {}

  get currentBackendJobId(): string {
    return this.backendJobId;
  }

  configure(config: TrainingConfig, layers: NetworkLayer[]): void {
    this.config = { ...config };
    this.layers = layers;
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

  async listCheckpoints(datasetId?: string): Promise<TrainingCheckpointSummary[]> {
    const query = datasetId ? `?datasetId=${encodeURIComponent(datasetId)}` : '';
    return this.api.request<TrainingCheckpointSummary[]>(`/api/training/checkpoints${query}`);
  }

  async testCheckpoint(checkpointId: number, request: Pick<BackendTrainingStartRequest, 'datasetId' | 'layers'>): Promise<TrainingTestResult> {
    const result = await this.api.request<TrainingTestResult>(`/api/training/checkpoints/${encodeURIComponent(String(checkpointId))}/test`, {
      method: 'POST',
      body: JSON.stringify(request)
    });
    const normalized: TrainingTestResult = {
      ...result,
      samples: (result.samples ?? []).map(sample => ({
        ...sample,
        imageUrl: sample.imageUrl ? this.normalizeResourceUrl(sample.imageUrl) : sample.imageUrl
      }))
    };
    this.testResult$.next(normalized);
    const accText = normalized.testAccuracy === null ? 'N/A' : `${(normalized.testAccuracy * 100).toFixed(1)}%`;
    this.log('info', `已使用 checkpoint 跑测试集：accuracy=${accText}, samples=${normalized.sampleCount}`);
    return normalized;
  }

  observeBackendJob(jobId: string): void {
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
    this.patchState({
      status: 'running',
      currentEpoch: 0,
      currentBatch: 0,
      totalBatches: 0,
      totalEpochs: 0,
      message: `Observing backend training: ${target}`
    });
    this.log('info', `加入训练观察：${target}`);
    this.connectWebSocket(`/api/training/stream?jobId=${encodeURIComponent(target)}`);
  }

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
    this.log('warn', 'Training stopped and reset.');
  }

  async reset(): Promise<void> {
    if (this.backendJobId) {
      this.history$.next([]);
      this.testResult$.next(null);
      await this.controlBackend('reset', 'Training reset.');
      return;
    }
    await this.stop();
    this.logs$.next([]);
  }

  ngOnDestroy(): void {
    this.clearTimer();
    this.closeSocket();
  }

  private connectWebSocket(streamUrl: string): void {
    this.closeSocket();
    const wsUrl = this.normalizeWebSocketUrl(streamUrl);
    const socket = new WebSocket(wsUrl);
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
      const result: TrainingTestResult = {
        jobId: message.jobId,
        testLoss: message.testLoss ?? null,
        testAccuracy: message.testAccuracy ?? null,
        sampleCount: message.sampleCount ?? 0,
        samples: (message.samples ?? []).map(sample => ({
          ...sample,
          imageUrl: sample.imageUrl ? this.normalizeResourceUrl(sample.imageUrl) : sample.imageUrl
        }))
      };
      this.testResult$.next(result);
      const accText = result.testAccuracy === null ? 'N/A' : `${(result.testAccuracy * 100).toFixed(1)}%`;
      this.log('info', `测试集评估完成：accuracy=${accText}, samples=${result.sampleCount}`);
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

  private closeSocket(): void {
    if (this.socket) {
      const socket = this.socket;
      this.socket = null;
      socket.onclose = null;
      socket.close();
    }
  }

  private normalizeWebSocketUrl(streamUrl: string): string {
    if (streamUrl.startsWith('ws://') || streamUrl.startsWith('wss://')) {
      return streamUrl;
    }
    if (streamUrl.startsWith('/')) {
      const base = this.api.baseUrl.replace(/^http/, 'ws');
      return `${base}${streamUrl}`;
    }
    return streamUrl;
  }

  private normalizeResourceUrl(url: string): string {
    if (!url || /^https?:\/\//i.test(url) || url.startsWith('data:')) return url;
    return `${this.api.baseUrl}${url.startsWith('/') ? url : '/' + url}`;
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
