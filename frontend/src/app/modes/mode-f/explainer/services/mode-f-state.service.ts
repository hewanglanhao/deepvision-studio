import { Injectable, computed, signal } from '@angular/core';
import { ModeFRnnEngine } from '../engine/mode-f-rnn-engine';
import { ModeFAssetsService } from './mode-f-assets.service';
import type {
  ModeFSequenceSample, ModeFTrainingConfig, ModeFVisualStatus, ModeFFocusArea,
  ModeFStepResult, ModeFNetworkPreset, ModeFDatasetPreset,
} from '../models/mode-f.types';

@Injectable({ providedIn: 'root' })
export class ModeFStateService {
  engine!: ModeFRnnEngine;

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

  // ---- topology data (for network visualization) ----

  readonly inputDim = computed(() => this.networkMeta()?.inputDim ?? 2);
  readonly hDim = computed(() => this.networkMeta()?.hiddenDim ?? 4);
  readonly outputDim = computed(() => this.networkMeta()?.outputDim ?? 2);
  readonly neuronCounts = computed(() => [this.inputDim(), this.hDim(), this.outputDim()]);

  /** Weight edges: W_xh (0→1), W_hy (1→2). W_hh shown as self-loop arrow on hidden box */
  readonly weightEdges = computed(() => {
    const step = this.currentStep();
    const _meta = this.networkMeta();
    const snap = step?.weightSnapshot;
    const engine = this.engine;
    const Wxh = snap?.WxhAfter ?? engine?.Wxh ?? [];
    const Why = snap?.WhyAfter ?? engine?.Why ?? [];
    const grad = step?.gradient;
    const edges: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number; weight: number; gradient?: number; type: 'wxh' | 'why' }[] = [];
    for (let i = 0; i < Wxh.length; i++) for (let j = 0; j < (Wxh[i]?.length ?? 0); j++)
      edges.push({ layerFrom: 0, neuronFrom: j, layerTo: 1, neuronTo: i, weight: Wxh[i][j], gradient: grad?.dWxh?.[i]?.[j], type: 'wxh' });
    for (let i = 0; i < Why.length; i++) for (let j = 0; j < (Why[i]?.length ?? 0); j++)
      edges.push({ layerFrom: 1, neuronFrom: j, layerTo: 2, neuronTo: i, weight: Why[i][j], gradient: grad?.dWhy?.[i]?.[j], type: 'why' });
    return edges;
  });

  /** Bias values for hidden and output layers */
  readonly biasValues = computed(() => {
    const _meta = this.networkMeta(); // track preset changes
    const step = this.currentStep();
    const snap = step?.weightSnapshot;
    const engine = this.engine;
    const bh = snap?.bhAfter ?? engine?.bh ?? [];
    const by_ = snap?.byAfter ?? engine?.by ?? [];
    const grad = step?.gradient;
    const result: { layerIdx: number; neuronIdx: number; bias: number; gradient?: number }[] = [];
    for (let i = 0; i < bh.length; i++)
      result.push({ layerIdx: 1, neuronIdx: i, bias: bh[i], gradient: grad?.dbh?.[i] });
    for (let i = 0; i < by_.length; i++)
      result.push({ layerIdx: 2, neuronIdx: i, bias: by_[i], gradient: grad?.dby?.[i] });
    return result;
  });

  /** Per-layer per-neuron activation values (for neuron circle coloring) */
  readonly neuronActivations = computed(() => {
    const step = this.currentStep();
    const inDim = this.inputDim();
    const hd = this.hDim();
    const od = this.outputDim();
    const inputs: number[] = new Array(inDim).fill(0);
    const hidden: number[] = new Array(hd).fill(0);
    const outputs: number[] = new Array(od).fill(0);
    if (step) {
      const lastState = step.forwardResult.states[step.timeSteps - 1];
      if (lastState) {
        // Input: use the last time step's input from the dataset
        const sample = this.currentDataset()[this.currentSampleIndex()];
        if (sample) {
          const lastInput = sample.inputs[sample.inputs.length - 1];
          if (lastInput) for (let i = 0; i < Math.min(inDim, lastInput.length); i++) inputs[i] = lastInput[i];
        }
        // Hidden: tanh activations
        for (let i = 0; i < Math.min(hd, lastState.hidden.length); i++) hidden[i] = lastState.hidden[i];
        // Output: softmax probabilities
        for (let i = 0; i < Math.min(od, lastState.output.length); i++) outputs[i] = lastState.output[i];
      }
    }
    return [inputs, hidden, outputs];
  });

  // ---- neuron selection ----

  readonly selectedNeuronRef = signal<{ layerIdx: number; neuronIdx: number } | null>(null);

  readonly selectedNeuron = computed(() => {
    const ref = this.selectedNeuronRef();
    if (!ref) return null;
    const acts = this.neuronActivations();
    const val = acts[ref.layerIdx]?.[ref.neuronIdx] ?? 0;
    const incoming = this.weightEdges().filter(e => e.layerTo === ref.layerIdx && e.neuronTo === ref.neuronIdx);
    const outgoing = this.weightEdges().filter(e => e.layerFrom === ref.layerIdx && e.neuronFrom === ref.neuronIdx);
    const bias = this.biasValues().find(b => b.layerIdx === ref.layerIdx && b.neuronIdx === ref.neuronIdx);
    const layerNames = ['输入层', '隐层 (tanh)', '输出层 (softmax)'];
    return { ...ref, activation: val, incoming, outgoing, bias, layerName: layerNames[ref.layerIdx] ?? '', layerType: layerNames[ref.layerIdx] ?? '' };
  });

  selectNeuron(layerIdx: number, neuronIdx: number): void {
    this.selectedNeuronRef.set({ layerIdx, neuronIdx });
    this.focusedArea.set('detail');
  }

  clearNeuronSelection(): void { this.selectedNeuronRef.set(null); }
}
