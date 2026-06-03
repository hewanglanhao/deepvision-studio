import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';
import {
  InferenceSampleItem,
  SingleInferenceActivation,
  SingleInferenceResult,
  TrainingCheckpointSummary,
  TrainingRuntimeService
} from '@shared/training/training-runtime.service';

@Component({
  selector: 'app-single-inference-page',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink, PlatformTopbarComponent],
  templateUrl: './single-inference-page.component.html',
  styleUrl: './single-inference-page.component.css'
})
export class SingleInferencePageComponent implements OnInit, OnDestroy {
  readonly Math = Math;
  authUser: AuthUser | null = null;
  checkpoints: TrainingCheckpointSummary[] = [];
  selectedCheckpointId: number | null = null;
  samples: InferenceSampleItem[] = [];
  selectedSample: InferenceSampleItem | null = null;
  inferenceResult: SingleInferenceResult | null = null;
  activeActivationOrder = 0;
  sampleDialogOpen = false;
  loadingCheckpoints = false;
  loadingSamples = false;
  inferring = false;
  error = '';
  private readonly subs = new Subscription();
  private playTimer: number | null = null;

  readonly topbarStatusPills = ['单样本推理', '逐层激活', 'Checkpoint'];

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
        this.selectedCheckpointId = null;
        this.samples = [];
        this.selectedSample = null;
        this.inferenceResult = null;
      }
    }));
    void this.authSvc.restoreSession();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    this.stopActivationPlayback();
  }

  get completedCheckpoints(): TrainingCheckpointSummary[] {
    return this.checkpoints.filter(checkpoint => this.isCompletedCheckpoint(checkpoint));
  }

  get selectedCheckpoint(): TrainingCheckpointSummary | null {
    return this.completedCheckpoints.find(checkpoint => checkpoint.id === this.selectedCheckpointId) ?? this.completedCheckpoints[0] ?? null;
  }

  get activeActivation(): SingleInferenceActivation | null {
    const activations = this.inferenceResult?.activations ?? [];
    return activations.find(item => item.order === this.activeActivationOrder) ?? activations[0] ?? null;
  }

  async loadCheckpoints(): Promise<void> {
    if (!this.authUser) return;
    this.loadingCheckpoints = true;
    this.error = '';
    try {
      this.checkpoints = await this.trainingSvc.listCheckpoints();
      if (!this.selectedCheckpointId || !this.completedCheckpoints.some(item => item.id === this.selectedCheckpointId)) {
        this.selectedCheckpointId = this.completedCheckpoints[0]?.id ?? null;
      }
      this.samples = [];
      this.selectedSample = null;
      this.inferenceResult = null;
    } catch (err) {
      this.error = err instanceof Error ? err.message : '加载 checkpoint 失败。';
    } finally {
      this.loadingCheckpoints = false;
    }
  }

  onCheckpointChange(): void {
    this.samples = [];
    this.selectedSample = null;
    this.inferenceResult = null;
    this.stopActivationPlayback();
  }

  async openSampleDialog(): Promise<void> {
    const checkpoint = this.selectedCheckpoint;
    if (!checkpoint) return;
    this.sampleDialogOpen = true;
    if (this.samples.length) return;
    this.loadingSamples = true;
    this.error = '';
    try {
      const result = await this.trainingSvc.listInferenceSamples(checkpoint.id, 72);
      this.samples = result.samples ?? [];
    } catch (err) {
      this.error = err instanceof Error ? err.message : '加载样本失败。';
    } finally {
      this.loadingSamples = false;
    }
  }

  chooseSample(sample: InferenceSampleItem): void {
    this.selectedSample = sample;
    this.sampleDialogOpen = false;
    this.inferenceResult = null;
    this.stopActivationPlayback();
  }

  async runInference(): Promise<void> {
    const checkpoint = this.selectedCheckpoint;
    const sample = this.selectedSample;
    if (!checkpoint || !sample) return;
    this.inferring = true;
    this.error = '';
    this.stopActivationPlayback();
    try {
      this.inferenceResult = await this.trainingSvc.inferCheckpointSample(checkpoint.id, sample.index);
      this.selectedSample = this.inferenceResult.sample ?? sample;
      this.activeActivationOrder = 0;
      this.startActivationPlayback();
    } catch (err) {
      this.error = err instanceof Error ? err.message : '单样本推理失败。';
    } finally {
      this.inferring = false;
    }
  }

  selectActivation(order: number): void {
    this.activeActivationOrder = order;
    this.stopActivationPlayback();
  }

  startActivationPlayback(): void {
    this.stopActivationPlayback();
    const count = this.inferenceResult?.activations?.length ?? 0;
    if (count <= 1) return;
    this.playTimer = window.setInterval(() => {
      this.activeActivationOrder = (this.activeActivationOrder + 1) % count;
    }, 950);
  }

  stopActivationPlayback(): void {
    if (this.playTimer !== null) {
      window.clearInterval(this.playTimer);
      this.playTimer = null;
    }
  }

  logout(): void {
    this.authSvc.logout();
  }

  checkpointLabel(checkpoint: TrainingCheckpointSummary): string {
    return `${checkpoint.name} · ${checkpoint.datasetName} · ${new Date(checkpoint.createdAt).toLocaleString()}`;
  }

  percent(value: number | null | undefined): string {
    return value === null || value === undefined ? 'N/A' : `${(value * 100).toFixed(1)}%`;
  }

  shapeLabel(shape: number[] | undefined): string {
    return shape?.length ? shape.join(' x ') : 'scalar';
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

  layerIcon(type: string): string {
    const icons: Record<string, string> = {
      input: '⬛',
      conv2d: '⊞',
      pool2d: '⊟',
      residual: '+',
      flatten: '≡',
      dense: '◉',
      activation: 'ƒ',
      dropout: '⊘',
      output: '▶'
    };
    return icons[type] ?? '□';
  }

  layerColor(type: string): string {
    const colors: Record<string, string> = {
      input: '#6366f1',
      conv2d: '#0ea5e9',
      pool2d: '#10b981',
      residual: '#14b8a6',
      flatten: '#f59e0b',
      dense: '#8b5cf6',
      activation: '#ec4899',
      dropout: '#94a3b8',
      output: '#ef4444'
    };
    return colors[type] ?? '#64748b';
  }

  sampleFeatureText(sample: InferenceSampleItem | null): string {
    if (!sample?.featurePreview?.length) return '';
    const values = sample.featurePreview.slice(0, 8).map(value => Number(value).toFixed(3)).join(', ');
    return `${values}${(sample.featureCount ?? 0) > 8 ? ' ...' : ''}`;
  }

  activationBars(activation: SingleInferenceActivation | null): Array<{ index: number; value: number; width: number }> {
    const values = activation?.preview?.values ?? [];
    const max = Math.max(0.000001, ...values.map(value => Math.abs(value)));
    return values.slice(0, 36).map((value, index) => ({
      index,
      value,
      width: Math.max(4, Math.abs(value) / max * 100)
    }));
  }

  private isCompletedCheckpoint(checkpoint: TrainingCheckpointSummary): boolean {
    return checkpoint.status !== 'stopped' && checkpoint.epoch >= checkpoint.totalEpochs;
  }
}
