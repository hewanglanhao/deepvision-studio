import { Injectable, computed, signal } from '@angular/core';
import { ModeFRnnEngine } from '../engine/mode-f-rnn-engine';
import { ModeFAssetsService } from './mode-f-assets.service';
import type {
  ModeFSequenceSample, ModeFTrainingConfig, ModeFVisualStatus, ModeFFocusArea,
  ModeFStepResult, ModeFNetworkPreset, ModeFDatasetPreset,
} from '../models/mode-f.types';

@Injectable({ providedIn: 'root' })
export class ModeFStateService {
  private engine!: ModeFRnnEngine;

  readonly status = signal<ModeFVisualStatus>('idle');
  readonly focusedArea = signal<ModeFFocusArea>('overview');
  readonly selectedPresetId = signal('echo-simple');

  readonly trainingConfig = signal<ModeFTrainingConfig>({ learningRate: 0.05, optimizer: 'adam', maxIterations: 800 });
  readonly currentIteration = signal(0);
  readonly currentSampleIndex = signal(0);
  readonly currentStep = signal<ModeFStepResult | null>(null);
  readonly stepHistory = signal<ModeFStepResult[]>([]);
  readonly isPlaying = signal(false);
  readonly playSpeed = signal(200);
  readonly lossHistory = signal<{ iteration: number; loss: number }[]>([]);
  readonly avgLossHistory = signal<{ iteration: number; loss: number; accuracy: number }[]>([]);
  readonly latestAccuracy = signal(0);
  readonly networkMeta = signal<ModeFNetworkPreset | null>(null);
  readonly datasetMeta = signal<ModeFDatasetPreset | null>(null);
  readonly currentDataset = signal<ModeFSequenceSample[]>([]);

  private playTimer: ReturnType<typeof setInterval> | null = null;

  constructor(private assets: ModeFAssetsService) {}

  loadPreset(id: string): void {
    const preset = this.assets.networkPresets.find(p => p.id === id);
    if (!preset) return;
    const dataset = this.assets.datasetPresets.find(d => d.id === preset.datasetId);
    if (!dataset) return;
    this.selectedPresetId.set(id);
    this.networkMeta.set(preset);
    this.datasetMeta.set(dataset);
    this.currentDataset.set(dataset.samples);
    this.engine = new ModeFRnnEngine(preset.inputDim, preset.hiddenDim, preset.outputDim);
    this.reset();
  }

  reset(): void { this.pause(); this.engine?.reset(); this.currentStep.set(null); this.stepHistory.set([]); this.currentIteration.set(0); this.lossHistory.set([]); this.avgLossHistory.set([]); this.latestAccuracy.set(0); this.status.set('ready'); }

  stepForward(): void {
    if (!this.engine) return;
    const dataset = this.currentDataset();
    const config = this.trainingConfig();
    const itr = this.currentIteration();
    const sample = dataset[this.currentSampleIndex()];
    const result = this.engine.trainStep(sample, config, itr);
    this.currentStep.set(result);
    this.currentIteration.set(itr + 1);
    const h = [...this.stepHistory(), result]; if (h.length > 500) h.shift(); this.stepHistory.set(h);
    const lh = [...this.lossHistory(), { iteration: itr, loss: result.loss }]; if (lh.length > 500) lh.shift(); this.lossHistory.set(lh);
    if ((itr + 1) % 10 === 0) this.computeAvg();
    const next = Math.floor(Math.random() * dataset.length); this.currentSampleIndex.set(next);
  }

  private computeAvg(): void {
    if (!this.engine) return;
    const dataset = this.currentDataset();
    let totalLoss = 0, correct = 0;
    for (const s of dataset) {
      const { finalPrediction } = this.engine.forward(s.inputs);
      const p = Math.max(finalPrediction[s.label], 1e-15); totalLoss += -Math.log(p);
      if (finalPrediction.indexOf(Math.max(...finalPrediction)) === s.label) correct++;
    }
    const alh = [...this.avgLossHistory(), { iteration: this.currentIteration(), loss: totalLoss / dataset.length, accuracy: correct / dataset.length }];
    if (alh.length > 200) alh.shift(); this.avgLossHistory.set(alh);
    this.latestAccuracy.set(correct / dataset.length);
  }

  togglePlay(): void { this.isPlaying() ? this.pause() : this.play(); }
  play(): void { if (this.isPlaying() || !this.engine) return; this.isPlaying.set(true); this.status.set('running'); const max = this.trainingConfig().maxIterations; const run = () => { if (this.currentIteration() >= max) { this.pause(); return; } this.stepForward(); }; run(); this.playTimer = setInterval(run, this.playSpeed()); }
  pause(): void { if (this.playTimer) { clearInterval(this.playTimer); this.playTimer = null; } this.isPlaying.set(false); this.status.set('paused'); }
  setTrainingConfig(p: Partial<ModeFTrainingConfig>): void { this.trainingConfig.set({ ...this.trainingConfig(), ...p }); }
  setPlaySpeed(ms: number): void { this.playSpeed.set(ms); if (this.isPlaying()) { this.pause(); this.play(); } }
  setFocusedArea(a: ModeFFocusArea): void { this.focusedArea.set(a); }
  setPreset(id: string): void { this.loadPreset(id); }

  readonly presetOptions = computed(() => this.assets.networkPresets);
  readonly datasetPresets = computed(() => this.assets.datasetPresets);
  readonly predictedLabel = computed(() => { const s = this.currentStep(); if (!s) return '—'; return this.datasetMeta()?.classLabels[s.predictedClass] ?? `类${s.predictedClass}`; });
  readonly trueLabel = computed(() => { const s = this.currentStep(); if (!s) return '—'; return this.datasetMeta()?.classLabels[s.trueClass] ?? `类${s.trueClass}`; });
}
