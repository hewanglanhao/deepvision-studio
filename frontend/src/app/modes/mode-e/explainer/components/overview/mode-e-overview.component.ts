import { Component, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModeEStateService, type SubStep } from '../../services/mode-e-state.service';

const LAYER_GAP = 240;
const NEURON_GAP = 60;
const NEURON_R = 20;
const PAD = 50;

/** Unique key for each flow dot animation instance, cycles to force restart */
let dotKey = 0;

@Component({
  selector: 'app-mode-e-overview',
  imports: [CommonModule],
  templateUrl: './mode-e-overview.component.html',
  styleUrl: './mode-e-overview.component.css',
})
export class ModeEOverviewComponent {
  readonly NEURON_R = NEURON_R;
  readonly PAD = PAD;

  /** Hovered edge for showing label on demand */
  readonly hoveredEdge = signal<{ layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number } | null>(null);

  /** Cursor position within the SVG (offsetX/offsetY from mousemove) */
  labelX = 0;
  labelY = 0;

  constructor(readonly s: ModeEStateService) {}

  onSvgMove(event: MouseEvent): void {
    const svg = event.currentTarget as SVGSVGElement;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    this.labelX = event.clientX - rect.left;
    this.labelY = event.clientY - rect.top;
  }

  onEdgeEnter(e: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number }): void {
    this.hoveredEdge.set(e);
  }

  onEdgeLeave(): void {
    this.hoveredEdge.set(null);
  }

  isEdgeHovered(e: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number }): boolean {
    const h = this.hoveredEdge();
    if (!h) return false;
    return h.layerFrom === e.layerFrom && h.neuronFrom === e.neuronFrom
        && h.layerTo === e.layerTo && h.neuronTo === e.neuronTo;
  }

  /** The label to show in the tooltip, or null if nothing to display */
  readonly hoverLabel = computed(() => {
    const h = this.hoveredEdge();
    if (!h) return null;
    // Only show if the edge belongs to the active layer pair
    if (!this.isActiveLayerPair(h)) return null;
    const match = this.edges().find(e =>
      e.layerFrom === h.layerFrom && e.neuronFrom === h.neuronFrom &&
      e.layerTo === h.layerTo && e.neuronTo === h.neuronTo
    );
    if (!match) return null;
    return this.edgeLabel(match);
  });

  /** Same layer pair as hovered edge but NOT the hovered edge itself */
  isEdgeDimmed(e: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number }): boolean {
    const h = this.hoveredEdge();
    if (!h) return false;
    return h.layerFrom === e.layerFrom && h.layerTo === e.layerTo
        && !(h.neuronFrom === e.neuronFrom && h.neuronTo === e.neuronTo);
  }

  /** Whether a training step has been computed (has activation data) */
  readonly hasStep = computed(() => !!this.s.currentStep());

  /** Format activation: show "—" when no data, actual value otherwise */
  fmtAct(li: number, ni: number): string {
    if (!this.hasStep()) return '—';
    const val = this.acts()[li]?.[ni] ?? 0;
    if (Math.abs(val) < 0.0001) return '0';
    if (Math.abs(val) < 0.01) return val.toFixed(4);
    return val.toFixed(3);
  }

  // ---- layout -----------------------------------------------------------

  readonly layers = computed(() => this.s.networkLayers());
  readonly counts = computed(() => this.s.neuronCounts());
  readonly acts = computed(() => this.s.neuronActivations());
  readonly edges = computed(() => this.s.weightEdges());
  readonly biases = computed(() => this.s.biasValues());
  readonly phase = computed(() => this.s.activePhase());
  readonly step = computed(() => this.s.currentStep());
  readonly selRef = computed(() => this.s.selectedNeuronRef());
  readonly sel = computed(() => this.s.selectedNeuron());
  readonly sub = computed(() => this.s.subStep());
  readonly maxNeurons = computed(() => Math.max(...this.counts(), 2));
  readonly svgW = computed(() => (this.layers().length - 1) * LAYER_GAP + 2 * PAD + 100);
  readonly svgH = computed(() => this.maxNeurons() * NEURON_GAP + 2 * PAD);

  nx(li: number): number { return PAD + 60 + li * LAYER_GAP; }
  ny(ni: number, li: number): number {
    const cnt = this.counts()[li] ?? 2;
    return PAD + 10 + (this.maxNeurons() - cnt) * NEURON_GAP / 2 + ni * NEURON_GAP + NEURON_GAP / 2;
  }

  // ---- flow animation keys ----------------------------------------------

  flowKey(edge: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number }): string {
    // Return stable key; animation restarts via CSS animation-iteration
    return `dot-${edge.layerFrom}-${edge.neuronFrom}-${edge.layerTo}-${edge.neuronTo}`;
  }

  // ---- sub-step: which edges are active? --------------------------------

  /** Whether a specific layer pair is currently the active one */
  isActiveLayerPair(edge: { layerFrom: number; layerTo: number }): boolean {
    const ss = this.sub();
    if (ss.type === 'forward' && ss.layerPair === edge.layerFrom) return true;
    if (ss.type === 'backward' && ss.layerPair === edge.layerFrom) return true;
    if (ss.type === 'update' && ss.layerIdx === edge.layerTo) return true;
    return false;
  }

  edgeOpacity(e: { layerFrom: number; layerTo: number }): number {
    return this.isActiveLayerPair(e) ? 1 : 0.45;
  }

  // Edge color: only active layer pair gets phase color, rest stay gray
  edgeStroke(e: { layerFrom: number; layerTo: number }): string {
    if (!this.isActiveLayerPair(e)) return '#cbd5e1';
    const p = this.phase();
    if (p === 'forward') return '#3b82f6';
    if (p === 'backward') return '#d97706';
    if (p === 'update') return '#10b981';
    return '#cbd5e1';
  }

  showFlowDots(e: { layerFrom: number; layerTo: number }): boolean {
    const ss = this.sub();
    if (ss.type === 'forward' && ss.layerPair === e.layerFrom) return true;
    if (ss.type === 'backward' && ss.layerPair === e.layerFrom) return true;
    return false;
  }

  flowDirection(): 'forward' | 'backward' {
    return this.sub().type === 'backward' ? 'backward' : 'forward';
  }

  // ---- neuron highlighting per sub-step ---------------------------------

  neuronHighlight(li: number): boolean {
    const ss = this.sub();
    if (ss.type === 'idle' || ss.type === 'done') return false;
    if (ss.type === 'forward') return li === ss.layerPair || li === ss.layerPair + 1;
    if (ss.type === 'loss') return li === this.layers().length - 1;
    if (ss.type === 'backward') return li === ss.layerPair || li === ss.layerPair + 1;
    if (ss.type === 'update') return li === ss.layerIdx;
    return false;
  }

  // ---- sub-step label ---------------------------------------------------

  subStepLabel(): string {
    const ss = this.sub();
    const layers = this.layers();
    const n = layers.length;
    switch (ss.type) {
      case 'idle': return '点击"单步"开始观察数据流动';
      case 'done': return '本轮训练完成';
      case 'loss': return `损失计算 — 比较预测与真实标签，得到 Loss=${this.fmt(this.step()?.loss ?? 0)}`;
      case 'forward': {
        const from = layers[ss.layerPair]?.name ?? `L${ss.layerPair}`;
        const to = layers[ss.layerPair + 1]?.name ?? `L${ss.layerPair + 1}`;
        return `前向传播：${from} → ${to}`;
      }
      case 'backward': {
        const from = layers[ss.layerPair + 1]?.name ?? `L${ss.layerPair + 1}`;
        const to = layers[ss.layerPair]?.name ?? `L${ss.layerPair}`;
        return `反向传播：梯度从 ${from} 回传至 ${to}`;
      }
      case 'update': {
        const name = layers[ss.layerIdx]?.name ?? `L${ss.layerIdx}`;
        return `参数更新：${name} 的权重和偏置`;
      }
    }
  }

  // ---- edge label positioning (staggered to avoid overlap) --------------

  labelPos(e: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number }): { mx: number; my: number; angle: number } {
    const x1 = this.nx(e.layerFrom);
    const y1 = this.ny(e.neuronFrom, e.layerFrom);
    const x2 = this.nx(e.layerTo);
    const y2 = this.ny(e.neuronTo, e.layerTo);
    const mx = (x1 + x2) / 2;
    const my = (y1 + y2) / 2;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const perpX = -dy / len;
    const perpY = dx / len;
    const cnt = this.counts()[e.layerFrom] ?? 2;
    const offset = (e.neuronFrom - e.neuronTo) * 10 + (e.neuronFrom + e.neuronTo - cnt) * 5;
    const rot = Math.atan2(dy, dx) * 180 / Math.PI;
    return { mx: mx + perpX * offset, my: my + perpY * offset, angle: rot };
  }

  // ---- formatting -------------------------------------------------------

  fmt(v: number): string {
    if (Math.abs(v) < 0.0001) return '0';
    if (Math.abs(v) < 0.01) return v.toFixed(4);
    return v.toFixed(3);
  }

  edgeLabel(e: { weight: number; gradient?: number; before?: number; after?: number }): string {
    if (this.phase() === 'backward' && e.gradient != null) return '∇' + this.fmt(e.gradient);
    if (this.sub().type === 'update' && e.after != null && e.before != null)
      return this.fmt(e.before) + '→' + this.fmt(e.after);
    return this.fmt(e.weight);
  }

  edgeLblColor(): string {
    const p = this.phase();
    if (p === 'forward') return '#2563eb';
    if (p === 'backward') return '#d97706';
    if (p === 'update') return '#7c3aed';
    return '#64748b';
  }

  // ---- neuron style -----------------------------------------------------

  nFill(li: number, ni: number): string {
    const val = this.acts()[li]?.[ni] ?? 0;
    const abs = Math.abs(val);
    const intensity = Math.min(abs / 3, 1);
    if (abs < 0.0001) return 'rgba(148,163,184,0.45)';
    if (val > 0) return `rgba(37,99,235,${0.45 + intensity * 0.45})`;
    return `rgba(220,38,38,${0.35 + intensity * 0.45})`;
  }

  nStroke(li: number, ni: number): string {
    return '#94a3b8';
  }

  nSW(li: number, ni: number): number {
    return 1;
  }

  // Highlight ring for active layer neurons (subtle colored glow)
  hlRingColor(li: number): string {
    if (!this.neuronHighlight(li)) return 'none';
    const ss = this.sub();
    if (ss.type === 'forward') return 'rgba(59,130,246,0.5)';
    if (ss.type === 'backward') return 'rgba(217,119,6,0.5)';
    if (ss.type === 'update') return 'rgba(16,185,129,0.5)';
    return 'none';
  }

  // Selected neuron glow color matches activation
  selGlowColor(li: number, ni: number): string {
    const val = this.acts()[li]?.[ni] ?? 0;
    if (val > 0) return 'rgba(59,130,246,0.4)';
    if (val < 0) return 'rgba(220,38,38,0.4)';
    return 'rgba(148,163,184,0.3)';
  }

  selectN(li: number, ni: number): void { this.s.selectNeuron(li, ni); }

  doStep(): void { this.s.startAnimatedStep(); }
  doNext(): void { this.s.advanceSubStep(); }
  doTogglePlay(): void { this.s.togglePlay(); }
  doReset(): void { this.s.reset(); }

  // Helper for template: safe access to union type
  subAs(): any { return this.sub(); }

  // ---- prediction compact -----------------------------------------------

  // ---- sub-step indicator dots (top bar) ---------------------------------

  readonly totalSubSteps = computed(() => {
    const n = this.layers().length;
    let c = (n - 1) * 2 + 1; // forward + backward + loss
    // update steps
    for (const l of this.layers()) {
      if (l.type === 'dense' || l.type === 'output') c++;
    }
    return c;
  });

  readonly currentSubStepIndex = computed(() => {
    const ss = this.sub();
    if (ss.type === 'idle' || ss.type === 'done') return -1;
    const n = this.layers().length;
    let idx = 0;
    // forward
    for (let i = 0; i < n - 1; i++) {
      if (ss.type === 'forward' && ss.layerPair === i) return idx;
      idx++;
    }
    if (ss.type === 'loss') return idx;
    idx++;
    // backward
    for (let i = n - 2; i >= 0; i--) {
      if (ss.type === 'backward' && ss.layerPair === i) return idx;
      idx++;
    }
    // update
    const layers = this.layers();
    for (let i = 0; i < n; i++) {
      if (layers[i].type === 'dense' || layers[i].type === 'output') {
        if (ss.type === 'update' && ss.layerIdx === i) return idx;
        idx++;
      }
    }
    return -1;
  });

  // ---- prediction compact -----------------------------------------------

  // ---- chart data -------------------------------------------------------

  readonly classColors = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6'];

  readonly curSample = computed(() => {
    const dataset = this.s.currentDataset();
    return dataset[this.s.currentSampleIndex()] ?? null;
  });

  readonly datasetPoints = computed(() => {
    const dataset = this.s.currentDataset();
    return dataset.map((s, i) => ({
      x: s.input[0], y: s.input[1], label: s.label,
      isCurrent: i === this.s.currentSampleIndex(),
    }));
  });

  readonly lossSvgPoints = computed(() => {
    const pts = this.s.lossHistory();
    if (pts.length < 2) return '';
    const maxLoss = Math.max(...pts.map(p => p.loss), 0.1);
    return pts.map((p, i) => {
      const x = (i / Math.max(pts.length - 1, 1)) * 200;
      const y = 80 - (p.loss / maxLoss) * 80;
      return `${x},${y}`;
    }).join(' ');
  });

  readonly maxLossLabel = computed(() => Math.max(...this.s.lossHistory().map(p => p.loss), 0.1).toFixed(3));

  readonly predPct = computed(() => {
    const s = this.step();
    if (!s?.predictions) return [];
    return s.predictions.map((p, i) => ({
      idx: i, pct: (p * 100).toFixed(1),
      label: this.s.datasetMeta()?.classLabels?.[i] ?? `类${i}`,
      correct: i === s.trueClass,
    }));
  });

  readonly lossVal = computed(() => this.step()?.loss?.toFixed(6) ?? '—');
  readonly predLabel = computed(() => this.s.predictedClassLabel());
  readonly trueLabel = computed(() => this.s.trueClassLabel());
}
