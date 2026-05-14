import { Injectable, computed, signal } from '@angular/core';
import type {
  ModeDBackpropStep,
  ModeDTrainingConfig,
  ModeDFocusArea,
  ModeDVisualStatus,
  ModeDNetworkPreset,
  ModeDDatasetPreset,
  ModeDDatasetSample,
} from '../models/mode-d.types';
import type { Connection } from '@shared/simulation/sim-models';
import { ModeDBackpropEngine } from '../engine/mode-d-backprop-engine';
import { ModeDAssetsService } from './mode-d-assets.service';

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
export class ModeDStateService {
  private engine = new ModeDBackpropEngine();

  // ---- writable signals -------------------------------------------------

  readonly status = signal<ModeDVisualStatus>('idle');
  readonly focusedArea = signal<ModeDFocusArea>('overview');
  readonly selectedPresetId = signal<string>('xor-mlp');

  readonly networkLayers = signal<LayerMeta[]>([]);
  readonly connections = signal<Connection[]>([]);
  readonly currentDataset = signal<ModeDDatasetSample[]>([]);
  readonly datasetMeta = signal<ModeDDatasetPreset | null>(null);
  readonly networkMeta = signal<ModeDNetworkPreset | null>(null);

  readonly trainingConfig = signal<ModeDTrainingConfig>({
    learningRate: 0.1, optimizer: 'adam', lossFunction: 'crossEntropy', maxIterations: 1000,
  });

  readonly currentStep = signal<ModeDBackpropStep | null>(null);
  readonly stepHistory = signal<ModeDBackpropStep[]>([]);
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

  constructor(private readonly assets: ModeDAssetsService) {}

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
    this.reset();
  }

  setTrainingConfig(partial: Partial<ModeDTrainingConfig>): void {
    this.trainingConfig.set({ ...this.trainingConfig(), ...partial });
  }

  setPlaySpeed(ms: number): void {
    this.playSpeed.set(ms);
    if (this.isPlaying()) { this.pause(); this.play(); }
  }

  setFocusedArea(area: ModeDFocusArea): void { this.focusedArea.set(area); }

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
  }

  /** Latest full-dataset accuracy */
  readonly latestAccuracy = signal(0);

  /** Save current avg-loss curve for optimizer comparison (called on training complete) */
  saveCurrentCurve(): void {
    const points = this.avgLossHistory();
    if (points.length === 0) return;
    const config = this.trainingConfig();
    const colors: Record<string, string> = { sgd: '#2563eb', momentum: '#7c3aed', adam: '#059669' };
    const curves = this.savedCurves().filter(c => c.label !== `${config.optimizer.toUpperCase()}`);
    const last = points[points.length - 1];
    curves.push({
      label: `${config.optimizer.toUpperCase()} (Acc ${(last.accuracy * 100).toFixed(0)}%)`,
      color: colors[config.optimizer] ?? '#94a3b8',
      points: [...points],
    });
    this.savedCurves.set(curves);
  }

  /** Clear saved comparison curves */
  clearSavedCurves(): void {
    this.savedCurves.set([]);
  }
  setPreset(presetId: string): void { this.loadPreset(presetId); }
  setActiveLayer(layerId: number | null): void {
    this.activeLayerId.set(layerId);
    if (layerId) this.focusedArea.set('detail');
  }
}
