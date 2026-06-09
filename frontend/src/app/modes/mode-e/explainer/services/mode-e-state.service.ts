import { Injectable, computed, signal } from '@angular/core';
import type {
  ModeEBackpropStep,
  ModeETrainingConfig,
  ModeEFocusArea,
  ModeEVisualStatus,
  ModeENetworkPreset,
  ModeEDatasetPreset,
  ModeEDatasetSample,
} from '../models/mode-e.types';
import type { Connection } from '@shared/simulation/sim-models';
import { ModeEBackpropEngine } from '../engine/mode-e-backprop-engine';
import { ModeEAssetsService } from './mode-e-assets.service';

// ---- sub-step animation state machine ------------------------------------

export type SubStep =
  | { type: 'idle' }
  | { type: 'forward'; layerPair: number }   // forward from layerPair to layerPair+1
  | { type: 'loss' }
  | { type: 'backward'; layerPair: number }  // backward from layerPair+1 to layerPair
  | { type: 'update'; layerIdx: number }     // update weights for layer layerIdx
  | { type: 'done' };

interface LayerMeta {
  id: number; type: string; name: string; params: Record<string, any>;
}

@Injectable({ providedIn: 'root' })
export class ModeEStateService {
  private engine = new ModeEBackpropEngine();

  // ---- writable signals -------------------------------------------------

  readonly status = signal<ModeEVisualStatus>('idle');
  readonly focusedArea = signal<ModeEFocusArea>('overview');
  readonly selectedPresetId = signal<string>('xor-mlp');

  readonly networkLayers = signal<LayerMeta[]>([]);
  readonly connections = signal<Connection[]>([]);
  readonly currentDataset = signal<ModeEDatasetSample[]>([]);
  readonly datasetMeta = signal<ModeEDatasetPreset | null>(null);
  readonly networkMeta = signal<ModeENetworkPreset | null>(null);

  readonly trainingConfig = signal<ModeETrainingConfig>({
    learningRate: 0.1, optimizer: 'adam', lossFunction: 'crossEntropy', maxIterations: 1000,
  });

  readonly currentStep = signal<ModeEBackpropStep | null>(null);
  readonly stepHistory = signal<ModeEBackpropStep[]>([]);
  readonly currentIteration = signal(0);
  readonly currentSampleIndex = signal(0);

  readonly activeLayerId = signal<number | null>(null);
  readonly selectedNeuronRef = signal<{ layerIdx: number; neuronIdx: number } | null>(null);

  // ---- sub-step animation -------------------------------------------------

  readonly subStep = signal<SubStep>({ type: 'idle' });
  readonly isAnimating = signal(false);
  readonly activePhase = signal<'forward' | 'loss' | 'backward' | 'update'>('forward');

  private pendingSubSteps = signal<SubStep[]>([]);
  private currentSubIdx = signal(0);

  /** Layer count for sub-step sequence generation */
  private get layerCount(): number { return this.networkLayers().length; }

  /** Build sub-step sequence from current network */
  private buildSubSteps(): SubStep[] {
    const n = this.layerCount;
    if (n < 2) return [];
    const steps: SubStep[] = [];
    for (let i = 0; i < n - 1; i++) steps.push({ type: 'forward', layerPair: i });
    steps.push({ type: 'loss' });
    for (let i = n - 2; i >= 0; i--) steps.push({ type: 'backward', layerPair: i });
    const layers = this.networkLayers();
    for (let i = 0; i < n; i++) {
      if (layers[i].type === 'dense' || layers[i].type === 'output') {
        steps.push({ type: 'update', layerIdx: i });
      }
    }
    return steps;
  }

  /** Compute full training step and freeze at first sub-step for manual inspection */
  startAnimatedStep(): void {
    if (this.isAnimating()) return;
    this.isAnimating.set(true);
    this.status.set('running');

    const layers = this.networkLayers();
    if (layers.length === 0) { this.isAnimating.set(false); return; }
    const dataset = this.currentDataset();
    const sample = dataset[this.currentSampleIndex()];
    if (!sample) { this.isAnimating.set(false); return; }

    const config = this.trainingConfig();
    const iteration = this.currentIteration();
    const step = this.engine.trainingStep(layers, sample.input, sample.label, config, iteration);

    this.currentStep.set(step);
    this.currentIteration.set(iteration + 1);
    const history = [...this.stepHistory(), step];
    if (history.length > 500) history.shift();
    this.stepHistory.set(history);
    if (step.loss != null) {
      const lh = [...this.lossHistory(), { iteration, loss: step.loss }];
      if (lh.length > 500) lh.shift();
      this.lossHistory.set(lh);
    }
    this.maybeRecordAvgLoss(iteration + 1);

    const nextIdx = Math.floor(Math.random() * dataset.length);
    this.currentSampleIndex.set(nextIdx);

    this.pendingSubSteps.set(this.buildSubSteps());
    this.currentSubIdx.set(0);
    if (this.pendingSubSteps().length > 0) {
      this.applySubStep(this.pendingSubSteps()[0]);
    } else {
      this.finishAnimation();
    }
  }

  /** Advance to next sub-step (called by user clicking "继续") */
  advanceSubStep(): void {
    if (!this.isAnimating()) return;
    this.currentSubIdx.update(n => n + 1);
    if (this.currentSubIdx() >= this.pendingSubSteps().length) {
      this.finishAnimation();
    } else {
      this.applySubStep(this.pendingSubSteps()[this.currentSubIdx()]);
    }
  }

  /** Whether there are more sub-steps remaining */
  readonly hasMoreSubSteps = computed(() => {
    return this.isAnimating() && this.currentSubIdx() < this.pendingSubSteps().length - 1;
  });

  /** Total sub-steps in current sequence */
  readonly totalPendingSubSteps = computed(() => this.pendingSubSteps().length);
  /** Remaining sub-steps after current */
  readonly remainingSubSteps = computed(() => Math.max(0, this.pendingSubSteps().length - this.currentSubIdx() - 1));

  private applySubStep(ss: SubStep): void {
    this.subStep.set(ss);
    this.activePhase.set(
      ss.type === 'loss' ? 'loss' :
      ss.type === 'update' ? 'update' :
      ss.type === 'backward' ? 'backward' : 'forward'
    );
  }

  private finishAnimation(): void {
    this.subStep.set({ type: 'done' });
    this.activePhase.set('update');
    this.isAnimating.set(false);
    this.pendingSubSteps.set([]);
    this.currentSubIdx.set(0);
    this.status.set('ready');
  }

  /** Fast-forward: reveal all sub-steps instantly (called during continuous play) */
  instantStep(): void {
    const layers = this.networkLayers();
    if (layers.length === 0) return;
    const dataset = this.currentDataset();
    const sample = dataset[this.currentSampleIndex()];
    if (!sample) return;
    const config = this.trainingConfig();
    const iteration = this.currentIteration();
    const step = this.engine.trainingStep(layers, sample.input, sample.label, config, iteration);
    this.currentStep.set(step);
    this.currentIteration.set(iteration + 1);
    this.activePhase.set('update');
    this.subStep.set({ type: 'done' });
    const history = [...this.stepHistory(), step];
    if (history.length > 500) history.shift();
    this.stepHistory.set(history);
    if (step.loss != null) {
      const lh = [...this.lossHistory(), { iteration, loss: step.loss }];
      if (lh.length > 500) lh.shift();
      this.lossHistory.set(lh);
    }
    this.maybeRecordAvgLoss(iteration + 1);
    const nextIdx = Math.floor(Math.random() * dataset.length);
    this.currentSampleIndex.set(nextIdx);
  }

  private animPlayTimer: ReturnType<typeof setInterval> | null = null;

  play(): void {
    if (this.isPlaying()) return;
    // Auto-reset if previous run completed
    if (this.currentIteration() >= this.trainingConfig().maxIterations) {
      this.reset();
    }
    this.isPlaying.set(true);
    this.status.set('running');
    const maxSteps = this.trainingConfig().maxIterations;
    const run = () => {
      if (this.currentIteration() >= maxSteps) {
        this.saveCurrentCurve();
        this.pause();
        return;
      }
      if (!this.isAnimating()) {
        this.instantStep();
      }
    };
    run(); // first step immediately
    this.animPlayTimer = setInterval(run, this.playSpeed());
  }

  togglePlay(): void {
    if (this.isPlaying()) { this.pause(); } else { this.play(); }
  }

  readonly isPlaying = signal(false);
  readonly playSpeed = signal(200);

  pause(): void {
    if (this.animPlayTimer) { clearInterval(this.animPlayTimer); this.animPlayTimer = null; }
    this.isPlaying.set(false);
    this.status.set('paused');
  }

  reset(): void {
    this.pause();
    this.isAnimating.set(false);
    this.pendingSubSteps.set([]);
    this.currentSubIdx.set(0);
    this.engine.reset();
    this.currentStep.set(null);
    this.stepHistory.set([]);
    this.currentIteration.set(0);
    this.lossHistory.set([]);
    this.avgLossHistory.set([]);
    this.latestAccuracy.set(0);
    this.gradientNormHistory.set([]);
    this.decisionBoundary.set(null);
    setTimeout(() => this.computeBoundary(), 100); // after layers reload
    this.subStep.set({ type: 'idle' });
    this.activePhase.set('forward');
    this.status.set('ready');
    const preset = this.networkMeta();
    if (preset) {
      this.networkLayers.set(preset.layers.map(l => ({
        id: l.id, type: l.type, name: l.name,
        params: JSON.parse(JSON.stringify((l as any).params ?? {})),
      })));
    }
  }

  // ---- computed signals (mostly same) ------------------------------------

  readonly presetOptions = computed(() => this.assets.networkPresets);
  readonly datasetPresets = computed(() => this.assets.datasetPresets);

  readonly readableStatus = computed(() => {
    const map: Record<string, string> = { idle: '就绪', ready: '已加载', running: '训练中', paused: '已暂停' };
    return map[this.status()] ?? this.status();
  });

  readonly totalTrainableParams = computed(() => {
    // Compute from architecture (neuron counts × input sizes), not from weight existence
    const counts = this.neuronCounts();
    let total = 0;
    for (let i = 1; i < this.networkLayers().length; i++) {
      const layer = this.networkLayers()[i];
      if (layer.type === 'dense' || layer.type === 'output') {
        const inCount = counts[i - 1]; // neurons from previous layer
        const outCount = counts[i];     // neurons in this layer
        total += outCount * inCount;    // weights
        total += outCount;              // biases
      }
    }
    return total;
  });

  readonly predictedClassLabel = computed(() => {
    const step = this.currentStep();
    if (!step || step.predictedClass == null) return '—';
    return this.datasetMeta()?.classLabels[step.predictedClass] ?? `类 ${step.predictedClass}`;
  });

  readonly trueClassLabel = computed(() => {
    const step = this.currentStep();
    if (!step || step.trueClass == null) return '—';
    return this.datasetMeta()?.classLabels[step.trueClass] ?? `类 ${step.trueClass}`;
  });

  readonly lossHistory = signal<{ iteration: number; loss: number }[]>([]);
  /** Periodically computed average loss over all samples — smooth trend line */
  readonly avgLossHistory = signal<{ iteration: number; loss: number; accuracy: number }[]>([]);

  /** Saved loss curves for optimizer comparison */
  readonly savedCurves = signal<{ label: string; color: string; points: { iteration: number; loss: number; accuracy?: number }[] }[]>([]);
  readonly gradientNormHistory = signal<{ iteration: number; norm: number }[]>([]);
  readonly decisionBoundary = signal<any>(null);

  // ---- neuron-level data ------------------------------------------------

  readonly neuronCounts = computed(() => {
    return this.networkLayers().map(l => {
      if (l.type === 'dense' || l.type === 'output') return l.params['units'] as number;
      if (l.type === 'input') {
        const w = l.params['width'] as number ?? 2;
        return w * (l.params['channels'] as number ?? 1);
      }
      const cache = this.currentStep()?.forwardCache;
      if (cache) {
        const entry = cache.find(e => e.layerId === l.id);
        if (entry?.output?.[0]) return entry.output[0].length;
      }
      return 2;
    });
  });

  readonly neuronActivations = computed(() => {
    const step = this.currentStep();
    const counts = this.neuronCounts();
    if (!step?.forwardCache) return counts.map(n => new Array(n).fill(0));
    return counts.map((n, li) => {
      const cache = step.forwardCache?.find(e => e.layerIndex === li);
      return cache?.output?.[0]?.slice(0, n) ?? new Array(n).fill(0);
    });
  });

  readonly weightEdges = computed(() => {
    const layers = this.networkLayers();
    const step = this.currentStep();
    const edges: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number; weight: number; gradient?: number; before?: number; after?: number }[] = [];
    for (let li = 0; li < layers.length - 1; li++) {
      const toLayer = layers[li + 1];
      if (toLayer.type !== 'dense' && toLayer.type !== 'output') continue;
      const W = toLayer.params['weights'] as number[][] | undefined;
      if (!W) continue;
      const snap = step?.parameterSnapshots.find(s => s.layerId === toLayer.id);
      const grad = step?.layerGradients.find(g => g.layerId === toLayer.id);
      for (let ni = 0; ni < W.length; ni++) {
        for (let nj = 0; nj < W[ni].length; nj++) {
          edges.push({
            layerFrom: li, neuronFrom: nj, layerTo: li + 1, neuronTo: ni,
            weight: W[ni][nj],
            gradient: grad?.weightGradients?.[ni]?.[nj],
            before: snap?.weightsBefore?.[ni]?.[nj],
            after: snap?.weightsAfter?.[ni]?.[nj],
          });
        }
      }
    }
    return edges;
  });

  readonly biasValues = computed(() => {
    const layers = this.networkLayers();
    const step = this.currentStep();
    const result: { layerIdx: number; neuronIdx: number; bias: number; gradient?: number; before?: number; after?: number }[] = [];
    for (let li = 0; li < layers.length; li++) {
      if (layers[li].type !== 'dense' && layers[li].type !== 'output') continue;
      const b = layers[li].params['bias'] as number[] | undefined;
      if (!b) continue;
      const snap = step?.parameterSnapshots.find(s => s.layerId === layers[li].id);
      const grad = step?.layerGradients.find(g => g.layerId === layers[li].id);
      for (let ni = 0; ni < b.length; ni++) {
        result.push({
          layerIdx: li, neuronIdx: ni, bias: b[ni],
          gradient: grad?.biasGradients?.[ni],
          before: snap?.biasBefore?.[ni], after: snap?.biasAfter?.[ni],
        });
      }
    }
    return result;
  });

  readonly selectedNeuron = computed(() => {
    const ref = this.selectedNeuronRef();
    if (!ref) return null;
    const acts = this.neuronActivations();
    const val = acts[ref.layerIdx]?.[ref.neuronIdx] ?? 0;
    const incoming = this.weightEdges().filter(e => e.layerTo === ref.layerIdx && e.neuronTo === ref.neuronIdx);
    const outgoing = this.weightEdges().filter(e => e.layerFrom === ref.layerIdx && e.neuronFrom === ref.neuronIdx);
    const bias = this.biasValues().find(b => b.layerIdx === ref.layerIdx && b.neuronIdx === ref.neuronIdx);
    const layers = this.networkLayers();
    return { ...ref, activation: val, incoming, outgoing, bias, layerName: layers[ref.layerIdx]?.name ?? '', layerType: layers[ref.layerIdx]?.type ?? '' };
  });

  selectNeuron(layerIdx: number, neuronIdx: number): void {
    this.selectedNeuronRef.set({ layerIdx, neuronIdx });
    this.focusedArea.set('detail');
  }

  clearNeuronSelection(): void { this.selectedNeuronRef.set(null); }

  // ---- misc ------------------------------------------------------------

  constructor(private readonly assets: ModeEAssetsService) {}

  loadPreset(presetId: string): void {
    const preset = this.assets.networkPresets.find(p => p.id === presetId);
    if (!preset) return;
    const dataset = this.assets.datasetPresets.find(d => d.id === preset.datasetId);
    if (!dataset) return;
    this.selectedPresetId.set(presetId);
    this.networkMeta.set(preset);
    this.datasetMeta.set(dataset);
    this.networkLayers.set(preset.layers.map(l => ({
      id: l.id, type: l.type, name: l.name,
      params: JSON.parse(JSON.stringify((l as any).params ?? {})),
    })));
    this.connections.set(preset.connections.map(c => ({ ...c })));
    this.currentDataset.set(dataset.samples.map(s => ({ input: [...s.input], label: s.label })));
    // Sync currentActivation from the first dense layer's activation
    const firstDense = preset.layers.find((l: any) => l.type === 'dense');
    if (firstDense) this.currentActivation.set((firstDense as any).params['activation'] ?? 'relu');
    this.reset();
  }

  setTrainingConfig(partial: Partial<ModeETrainingConfig>): void {
    this.trainingConfig.set({ ...this.trainingConfig(), ...partial });
  }

  setPlaySpeed(ms: number): void {
    this.playSpeed.set(ms);
    if (this.isPlaying()) { this.pause(); this.play(); }
  }

  setFocusedArea(area: ModeEFocusArea): void { this.focusedArea.set(area); }

  readonly currentActivation = signal('relu');

  setActivation(act: string): void {
    this.currentActivation.set(act);
    // Reset first to get fresh random weights
    this.reset();
    // Then patch the activation on the freshly loaded layers
    const layers = this.networkLayers();
    for (const l of layers) {
      if (l.type === 'dense' || l.type === 'output') {
        l.params['activation'] = l.type === 'output' ? 'softmax' : act;
      }
    }
    // Force signal update since we mutated in place
    this.networkLayers.set([...layers]);
  }

  /** Compute average loss + accuracy over all samples every 25 steps */
  private maybeRecordAvgLoss(itr: number): void {
    if (itr % 25 !== 0) return;
    const layers = this.networkLayers();
    if (layers.length === 0) return;
    const dataset = this.currentDataset();
    if (dataset.length === 0) return;
    let totalLoss = 0;
    let correct = 0;
    const lossFn = this.trainingConfig().lossFunction;
    for (const sample of dataset) {
      const { output } = this.engine.forwardPass(layers, sample.input);
      const { loss } = this.engine.computeLoss(output, sample.label, lossFn);
      totalLoss += loss;
      if (output.indexOf(Math.max(...output)) === sample.label) correct++;
    }
    const avg = totalLoss / dataset.length;
    const acc = correct / dataset.length;
    const alh = [...this.avgLossHistory(), { iteration: itr, loss: avg, accuracy: acc }];
    if (alh.length > 200) alh.shift();
    this.avgLossHistory.set(alh);
    this.latestAccuracy.set(acc);
    this.computeBoundary();
  }

  /** Latest full-dataset accuracy */
  readonly latestAccuracy = signal(0);

  private computeBoundary(): void {
    const layers = this.networkLayers();
    if (layers.length < 2) return;
    const dataset = this.currentDataset();
    if (dataset.length === 0) return;
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const s of dataset) {
      if (s.input[0] < xMin) xMin = s.input[0];
      if (s.input[0] > xMax) xMax = s.input[0];
      if (s.input[1] < yMin) yMin = s.input[1];
      if (s.input[1] > yMax) yMax = s.input[1];
    }
    const xPad = Math.max((xMax - xMin) * 0.08, 0.02);
    const yPad = Math.max((yMax - yMin) * 0.08, 0.02);
    try {
      const b = this.engine.computeDecisionBoundary(layers, 50, [xMin - xPad, xMax + xPad], [yMin - yPad, yMax + yPad]);
      this.decisionBoundary.set(b);
    } catch { /* ignore */ }
  }

  /** Save current avg-loss curve for optimizer comparison (called on training complete) */
  saveCurrentCurve(): void {
    const points = this.avgLossHistory();
    if (points.length === 0) return;
    const config = this.trainingConfig();
    const act = this.currentActivation();
    const combo = `${config.optimizer}+${act}`;
    const colorPalette = ['#2563eb', '#7c3aed', '#059669', '#d97706', '#dc2626', '#0891b2', '#ca8a04', '#be185d'];
    const usedColors = new Set(this.savedCurves().map(c => c.color));
    const color = colorPalette.find(c => !usedColors.has(c)) ?? colorPalette[this.savedCurves().length % colorPalette.length];
    const last = points[points.length - 1];
    const curves = this.savedCurves().filter(c => c.label.split(' (')[0] !== combo);
    curves.push({
      label: `${combo} (Acc ${(last.accuracy * 100).toFixed(0)}%)`,
      color,
      points: [...points],
    });
    this.savedCurves.set(curves);
  }

  deleteSavedCurve(idx: number): void {
    const curves = [...this.savedCurves()];
    curves.splice(idx, 1);
    this.savedCurves.set(curves);
  }

  clearSavedCurves(): void {
    this.savedCurves.set([]);
  }
  setPreset(presetId: string): void { this.loadPreset(presetId); }
  setActiveLayer(layerId: number | null): void {
    this.activeLayerId.set(layerId);
    if (layerId) this.focusedArea.set('detail');
  }
}
