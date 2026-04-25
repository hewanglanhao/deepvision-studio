/**
 * TrainingRuntimeService
 *
 * 当前实现：MOCK 模式 — 使用 SimEngine 的数学公式模拟训练曲线。
 * 未来接入：将 startMock() 替换为 connectWebSocket(url) 或 callRestApi(url)，
 *           后端为 Spring Boot + WebSocket / SSE 推送每个 epoch 的 MetricPoint。
 *
 * 接口边界：
 *   REST  POST /api/training/start   { config, layers }  → { jobId }
 *   WS    ws://host/api/training/stream?jobId=xxx        → MetricPoint stream
 */
import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject, Subject } from 'rxjs';
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

@Injectable({ providedIn: 'root' })
export class TrainingRuntimeService implements OnDestroy {
  private timer: ReturnType<typeof setInterval> | null = null;

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

  configure(config: TrainingConfig, layers: NetworkLayer[]): void {
    this.config = { ...config };
    this.layers = layers;
  }

  start(): void {
    const s = this.state$.value;
    if (s.status === 'running') return;
    this.patchState({ status: 'running', message: '[MOCK] Training started.' });
    this.log('info', `[MOCK] Start — optimizer=${this.config.optimizer}, lr=${this.config.learningRate}, epochs=${this.config.totalEpochs}`);
    this.startMock();
  }

  pause(): void {
    this.clearTimer();
    if (this.state$.value.status === 'running') {
      this.patchState({ status: 'paused', message: 'Paused.' });
      this.log('info', 'Training paused.');
    }
  }

  stop(): void {
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
    this.log('warn', 'Training stopped and reset.');
  }

  reset(): void {
    this.stop();
    this.logs$.next([]);
  }

  ngOnDestroy(): void {
    this.clearTimer();
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
