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
  /** Clicked edge for showing specific formula */
  readonly clickedEdge = signal<{ layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number } | null>(null);
  /** Clicked neuron for showing its computation */
  readonly clickedNeuron = signal<{ layerIdx: number; neuronIdx: number } | null>(null);

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

  onEdgeClick(e: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number }): void {
    const current = this.clickedEdge();
    if (current?.layerFrom === e.layerFrom && current?.neuronFrom === e.neuronFrom &&
        current?.layerTo === e.layerTo && current?.neuronTo === e.neuronTo) {
      this.clickedEdge.set(null); // toggle off
    } else {
      this.clickedEdge.set(e);
      this.clickedNeuron.set(null);
    }
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

  selectN(li: number, ni: number): void {
    this.s.selectNeuron(li, ni);
    const cur = this.clickedNeuron();
    if (cur?.layerIdx === li && cur?.neuronIdx === ni) {
      this.clickedNeuron.set(null); // toggle off
    } else {
      this.clickedNeuron.set({ layerIdx: li, neuronIdx: ni });
      this.clickedEdge.set(null);
    }
  }

  /** Formula line showing the current neuron's computation */
  private fmtH(v: number, color: string): string { return `<span style="color:${color};font-weight:600">${this.fmt(v)}</span>`; }

  /** Get the target neuron index for formula display (clicked > sub-step default) */
  private getFormulaNeuron(targetLayer: number): number {
    const cn = this.clickedNeuron();
    if (cn && cn.layerIdx === targetLayer) return cn.neuronIdx;
    const ce = this.clickedEdge();
    if (ce && ce.layerTo === targetLayer) return ce.neuronTo;
    return 0; // default: first neuron
  }

  readonly formulaLine = computed(() => {
    const ss = this.sub();
    const step = this.step();
    if (!step || ss.type === 'idle' || ss.type === 'done') return null;
    const layers = this.layers();
    const acts = this.acts();
    const ce = this.clickedEdge();

    // Edge-level formula (clicked edge takes priority) — look up full data from edges()
    const edgeData = ce ? this.edges().find(e =>
      e.layerFrom === ce.layerFrom && e.neuronFrom === ce.neuronFrom &&
      e.layerTo === ce.layerTo && e.neuronTo === ce.neuronTo
    ) : null;

    if (edgeData && ss.type === 'forward') {
      const w = this.fmtH(edgeData.weight, '#2563eb');
      const a = this.fmtH(acts[edgeData.layerFrom]?.[edgeData.neuronFrom] ?? 0, '#7c3aed');
      const prod = this.fmtH(edgeData.weight * (acts[edgeData.layerFrom]?.[edgeData.neuronFrom] ?? 0), '#d97706');
      return `<span style="color:#64748b">前向:</span> W<sub>${edgeData.neuronTo},${edgeData.neuronFrom}</sub>×a<sub>${edgeData.neuronFrom}</sub> = ${w}×${a} = ${prod}`;
    }
    if (edgeData && ss.type === 'backward') {
      const g = this.fmtH(edgeData.gradient ?? 0, '#2563eb');
      const aPrev = this.fmtH(acts[edgeData.layerFrom]?.[edgeData.neuronFrom] ?? 0, '#7c3aed');
      const dW = this.fmtH((acts[edgeData.layerFrom]?.[edgeData.neuronFrom] ?? 0) * (edgeData.gradient ?? 0), '#d97706');
      return `<span style="color:#64748b">反向:</span> dW<sub>${edgeData.neuronTo},${edgeData.neuronFrom}</sub> = a<sub>${edgeData.neuronFrom}</sub>×dZ<sub>${edgeData.neuronTo}</sub> = ${aPrev}×${g} = ${dW}`;
    }
    if (edgeData && ss.type === 'update') {
      const before = this.fmtH(edgeData.before ?? 0, '#dc2626');
      const after = this.fmtH(edgeData.after ?? 0, '#059669');
      const delta = this.fmtH(Math.abs((edgeData.after ?? 0) - (edgeData.before ?? 0)), '#d97706');
      return `<span style="color:#64748b">更新:</span> W<sub>${edgeData.neuronTo},${edgeData.neuronFrom}</sub> = ${before} <span style="color:#94a3b8">→</span> ${after} <span style="color:#94a3b8">(Δ=${delta})</span>`;
    }

    // Neuron-level formula (clicked neuron or sub-step default)
    if (ss.type === 'loss') {
      const preds = step.predictions;
      if (!preds) return null;
      const tc = step.trueClass ?? 0;
      const probsHTML = preds.map((p, i) => `<span style="color:${i===tc?'#059669':'#64748b'}">${(p*100).toFixed(1)}%</span>`).join(', ');
      const lossVal = this.fmtH(step.loss ?? 0, '#d97706');
      return `<span style="color:#64748b">损失:</span> Softmax → [${probsHTML}] <span style="color:#94a3b8">真实=${tc}</span> −ln(p) = ${lossVal}`;
    }

    if (ss.type === 'forward') {
      const li = ss.layerPair + 1;
      const ni = this.getFormulaNeuron(li);
      const cache = step.forwardCache?.find(c => c.layerIndex === li);
      if (!cache?.preActivation) return null;
      // Use before-update weights to match the cached activations
      const snap = step.parameterSnapshots.find(s => s.layerId === layers[li]?.id);
      const wBefore = snap?.weightsBefore ?? layers[li]?.params?.['weights'] as number[][] | undefined;
      if (!wBefore?.[ni]) return null;
      const incoming = this.edges().filter(e => e.layerTo === li && e.neuronTo === ni);
      if (incoming.length === 0) return null;
      const bBefore = snap?.biasBefore?.[ni] ?? layers[li]?.params?.['bias']?.[ni] as number ?? 0;
      const terms = incoming.map((e, idx) => {
        const a = acts[e.layerFrom]?.[e.neuronFrom] ?? 0;
        const w = wBefore[ni]?.[idx] ?? e.weight;
        return `<span style="color:#2563eb;font-weight:600">${this.fmt(w)}</span><span style="color:#94a3b8">×</span><span style="color:#7c3aed;font-weight:600">${this.fmt(a)}</span>`;
      });
      const sum = wBefore[ni].reduce((s: number, w: number, idx: number) => s + w * (acts[li-1]?.[idx] ?? 0), 0) + bBefore;
      const actName = layers[li]?.params?.['activation'] ?? '线性';
      const actResult = actName === 'relu' ? Math.max(0, sum) : actName === 'sigmoid' ? 1 / (1 + Math.exp(-sum)) : actName === 'tanh' ? Math.tanh(sum) : sum;
      const biasHTML = bBefore !== 0 ? ` <span style="color:#94a3b8">+</span> <span style="color:#d97706;font-weight:600">${this.fmt(bBefore)}</span>` : '';
      return `<span style="color:#64748b">前向:</span> ${layers[li]?.name ?? 'L'+li}·神经元${ni}: (${terms.join(' <span style="color:#94a3b8">+</span> ')})${biasHTML} <span style="color:#94a3b8">=</span> <span style="color:#d97706;font-weight:600">${this.fmt(sum)}</span> <span style="color:#94a3b8">→ ${actName} →</span> <span style="color:#d97706;font-weight:600">${this.fmt(actResult)}</span>`;
    }

    if (ss.type === 'backward') {
      const li = ss.layerPair + 1;
      const ni = this.getFormulaNeuron(li);
      const grad = step.layerGradients.find(g => g.layerId === layers[li]?.id);
      if (!grad?.weightGradients?.[ni]?.[0]) return null;
      const cache = step.forwardCache?.find(c => c.layerIndex === li);
      const aPrev = cache?.input[0]?.[0] ?? 0;
      const dW = grad.weightGradients[ni][0];
      const dZ = grad.biasGradients?.[ni] ?? 0;
      const aPrevH = this.fmtH(aPrev, '#7c3aed');
      const dZH = this.fmtH(dZ, '#2563eb');
      const dWH = this.fmtH(dW, '#d97706');
      return `<span style="color:#64748b">反向:</span> ${layers[li]?.name ?? 'L'+li}·神经元${ni}: dW<sub>${ni},0</sub> = a<sub>0</sub>×dZ<sub>${ni}</sub> = ${aPrevH}×${dZH} = ${dWH}`;
    }

    if (ss.type === 'update') {
      const li = ss.layerIdx;
      const ni = this.getFormulaNeuron(li);
      const snap = step.parameterSnapshots.find(s => s.layerId === layers[li]?.id);
      const wBefore = snap?.weightsBefore?.[ni]?.[0];
      const wAfter = snap?.weightsAfter?.[ni]?.[0];
      if (wBefore == null || wAfter == null) return null;
      return `<span style="color:#64748b">更新:</span> ${layers[li]?.name ?? 'L'+li}·神经元${ni}: W<sub>${ni},0</sub> = <span style="color:#dc2626;font-weight:600">${this.fmt(wBefore)}</span> <span style="color:#94a3b8">→</span> <span style="color:#059669;font-weight:600">${this.fmt(wAfter)}</span> <span style="color:#94a3b8">(Δ=${this.fmtH(Math.abs(wAfter-wBefore),'#d97706')})</span>`;
    }

    return null;
  });

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
