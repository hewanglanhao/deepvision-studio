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
  fmtZ(li: number, ni: number): string {
    const step = this.step();
    if (!step) return '—';
    const cache = step.forwardCache?.find(e => e.layerIndex === li);
    const val = cache?.preActivation?.[0]?.[ni];
    if (val == null) return '—';
    return this.fmt(val);
  }

  fmtAct(li: number, ni: number): string {
    const step = this.step();
    if (!step) return '—';
    // Show '—' for layers not yet computed in incremental mode
    const cache = step.forwardCache?.find(e => e.layerIndex === li);
    if (!cache?.output?.[0]) return '—';
    const val = cache.output[0][ni] ?? 0;
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
    return this.isActiveLayerPair(e) ? 1 : 0.7;
  }

  // Edge color: only active layer pair gets phase color, rest stay gray
  edgeStroke(e: { layerFrom: number; layerTo: number }): string {
    if (!this.isActiveLayerPair(e)) return '#94a3b8';
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
    const dx = x2 - x1;
    const dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const nx = -dy / len, ny = dx / len;  // unit normal
    const mx = (x1 + x2) / 2 - nx * 8;
    const my = (y1 + y2) / 2 - ny * 8;
    const angle = Math.atan2(dy, dx) * 180 / Math.PI;
    return { mx, my, angle };
  }

  // ---- formatting -------------------------------------------------------

  fmt(v: number): string {
    if (Math.abs(v) < 0.0001) return '0';
    if (Math.abs(v) < 0.01) return v.toFixed(4);
    return v.toFixed(3);
  }

  edgeLabel(e: { weight: number; gradient?: number; before?: number; after?: number }): string {
    const ss = this.sub();
    if (ss.type === 'backward' && e.gradient != null) return '∇' + this.fmt(e.gradient);
    if (ss.type === 'update' && e.after != null && e.before != null)
      return this.fmt(e.before) + '→' + this.fmt(e.after);
    // During forward animation, show pre-update weight to match cached activations
    if (ss.type === 'forward' && e.before != null) return this.fmt(e.before);
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

  /** Format a value wrapped in a CSS class span (for innerHTML formula content) */
  private fmtV(v: number, cssClass: string): string { return `<span class="${cssClass}">${this.fmt(v)}</span>`; }

  /** Get the target neuron index for formula display (clicked > sub-step default) */
  private getFormulaNeuron(targetLayer: number): number {
    const cn = this.clickedNeuron();
    if (cn && cn.layerIdx === targetLayer) return cn.neuronIdx;
    const ce = this.clickedEdge();
    if (ce && ce.layerTo === targetLayer) return ce.neuronTo;
    return 0; // default: first neuron
  }

  /** Card header metadata: phase badge + descriptive title */
  readonly formulaHeader = computed(() => {
    const step = this.step();
    const cn = this.clickedNeuron();
    const ce = this.clickedEdge();
    if (!step) return null;
    const ss = this.sub();
    const hasClick = cn || ce;
    if (!hasClick && (ss.type === 'idle' || ss.type === 'done')) return null;
    const layers = this.layers();

    // Edge-level formulas
    if (ce && ss.type === 'forward') {
      return { phase: 'forward' as const, phaseLabel: '前向传播', title: `W · a  (边 N${ce.neuronFrom} → N${ce.neuronTo})` };
    }
    if (ce && ss.type === 'backward') {
      return { phase: 'backward' as const, phaseLabel: '反向传播', title: `∂W / ∂a  (边 N${ce.neuronFrom} → N${ce.neuronTo})` };
    }
    if (ce && ss.type === 'update') {
      return { phase: 'update' as const, phaseLabel: '参数更新', title: `W 更新 (边 N${ce.neuronFrom} → N${ce.neuronTo})` };
    }

    // Neuron clicked outside animation
    const li = cn && (ss.type === 'idle' || ss.type === 'done') ? cn.layerIdx
      : ss.type === 'forward' ? ss.layerPair + 1
      : ss.type === 'backward' ? ss.layerPair + 1
      : ss.type === 'update' ? ss.layerIdx
      : -1;

    if (cn && (ss.type === 'idle' || ss.type === 'done')) {
      return { phase: 'neuron' as const, phaseLabel: '神经元', title: `${layers[cn.layerIdx]?.name ?? 'L' + cn.layerIdx} · 神经元 #${cn.neuronIdx} 激活计算` };
    }

    // Sub-step formulas
    if (ss.type === 'loss') {
      return { phase: 'loss' as const, phaseLabel: '损失计算', title: '交叉熵损失 (Cross-Entropy Loss)' };
    }
    if (ss.type === 'forward' && li >= 0) {
      const ni = this.getFormulaNeuron(li);
      return { phase: 'forward' as const, phaseLabel: '前向传播', title: `${layers[li]?.name ?? 'L' + li} · 神经元 #${ni} 加权和计算` };
    }
    if (ss.type === 'backward' && li >= 0) {
      const ni = this.getFormulaNeuron(li);
      return { phase: 'backward' as const, phaseLabel: '反向传播', title: `${layers[li]?.name ?? 'L' + li} · 神经元 #${ni} 梯度推导` };
    }
    if (ss.type === 'update' && li >= 0) {
      const ni = this.getFormulaNeuron(li);
      return { phase: 'update' as const, phaseLabel: '参数更新', title: `${layers[li]?.name ?? 'L' + li} · 神经元 #${ni} 权重更新` };
    }

    return null;
  });

  readonly formulaLine = computed(() => {
    const step = this.step();
    const cn = this.clickedNeuron();
    const ce = this.clickedEdge();
    // Show formula if: sub-step active, OR user clicked a neuron/edge
    if (!step) return null;
    const ss = this.sub();
    const hasClick = cn || ce;
    if (!hasClick && (ss.type === 'idle' || ss.type === 'done')) return null;
    const layers = this.layers();
    const acts = this.acts();

    // Edge-level formula (clicked edge takes priority) — look up full data from edges()
    const edgeData = ce ? this.edges().find(e =>
      e.layerFrom === ce.layerFrom && e.neuronFrom === ce.neuronFrom &&
      e.layerTo === ce.layerTo && e.neuronTo === ce.neuronTo
    ) : null;

    if (edgeData && ss.type === 'forward') {
      const w = this.fmtV(edgeData.before ?? edgeData.weight, 'fv-w');
      const a = this.fmtV(acts[edgeData.layerFrom]?.[edgeData.neuronFrom] ?? 0, 'fv-a');
      const prod = this.fmtV((edgeData.before ?? edgeData.weight) * (acts[edgeData.layerFrom]?.[edgeData.neuronFrom] ?? 0), 'fv-r');
      return `<span class="fv-lb">前向:</span> W<sub>${edgeData.neuronTo},${edgeData.neuronFrom}</sub><span class="fv-op">×</span>a<sub>${edgeData.neuronFrom}</sub> <span class="fv-op">=</span> ${w}<span class="fv-op">×</span>${a} <span class="fv-op">=</span> ${prod}`;
    }
    if (edgeData && ss.type === 'backward') {
      const g = this.fmtV(edgeData.gradient ?? 0, 'fv-w');
      const aPrev = this.fmtV(acts[edgeData.layerFrom]?.[edgeData.neuronFrom] ?? 0, 'fv-a');
      const dW = this.fmtV((acts[edgeData.layerFrom]?.[edgeData.neuronFrom] ?? 0) * (edgeData.gradient ?? 0), 'fv-r');
      return `<span class="fv-lb">反向:</span> dW<sub>${edgeData.neuronTo},${edgeData.neuronFrom}</sub> <span class="fv-op">=</span> a<sub>${edgeData.neuronFrom}</sub><span class="fv-op">×</span>dZ<sub>${edgeData.neuronTo}</sub> <span class="fv-op">=</span> ${aPrev}<span class="fv-op">×</span>${g} <span class="fv-op">=</span> ${dW}`;
    }
    if (edgeData && ss.type === 'update') {
      const before = this.fmtV(edgeData.before ?? edgeData.weight, 'fv-bf');
      const after = this.fmtV(edgeData.after ?? edgeData.weight, 'fv-af');
      const delta = this.fmtV(Math.abs((edgeData.after ?? edgeData.weight) - (edgeData.before ?? edgeData.weight)), 'fv-r');
      return `<span class="fv-lb">更新:</span> W<sub>${edgeData.neuronTo},${edgeData.neuronFrom}</sub> <span class="fv-op">=</span> ${before} <span class="fv-op">→</span> ${after} <span class="fv-op">(Δ=</span>${delta}<span class="fv-op">)</span>`;
    }

    // Neuron-level formula (clicked neuron or sub-step default)
    // When a neuron is clicked, always show its forward formula (even outside animation)
    if (cn && (ss.type === 'idle' || ss.type === 'done')) {
      const li = cn.layerIdx;
      const cache = step.forwardCache?.find(c => c.layerIndex === li);
      if (!cache?.preActivation || layers[li]?.type === 'input') return null;
      const ni = cn.neuronIdx;
      const incoming = this.edges().filter(e => e.layerTo === li && e.neuronTo === ni);
      if (incoming.length === 0) return null;
      const bias = this.biases().find(b => b.layerIdx === li && b.neuronIdx === ni)?.bias ?? 0;
      const termRows = incoming.map((e, idx) => {
        const a = acts[e.layerFrom]?.[e.neuronFrom] ?? 0;
        const prod = e.weight * a;
        return `<div class="fv-term-row"><span class="fv-tl">w<sub>${idx}</sub>·a<sub>${idx}</sub></span><span class="fv-w">${this.fmt(e.weight)}</span><span class="fv-op">×</span><span class="fv-a">${this.fmt(a)}</span><span class="fv-op">=</span><span class="fv-st">${this.fmt(prod)}</span></div>`;
      });
      const sum = incoming.reduce((s, e) => s + e.weight * (acts[e.layerFrom]?.[e.neuronFrom] ?? 0), 0) + bias;
      const actName = layers[li]?.params?.['activation'] ?? '线性';
      const actResult = actName === 'relu' ? Math.max(0, sum) : actName === 'sigmoid' ? 1 / (1 + Math.exp(-sum)) : actName === 'tanh' ? Math.tanh(sum) : sum;
      const biasRow = `<div class="fv-term-row fv-bias-row"><span class="fv-tl">+ b</span><span class="fv-r">${this.fmt(bias)}</span></div>`;
      const sumRow = `<div class="fv-term-row fv-sum-row"><span class="fv-tl">Z</span><span class="fv-r">${this.fmt(sum)}</span></div>`;
      const actLine = `<div class="fv-ar"><span class="ar-op">${actName} →</span> <span class="ar-val">${this.fmt(actResult)}</span></div>`;
      return `<div class="fv-section-label">加权和计算</div><div class="fv-terms">${termRows.join('')}</div><div class="fv-sep"></div>${biasRow}${sumRow}${actLine}`;
    }

    if (ss.type === 'loss') {
      const preds = step.predictions;
      if (!preds) return null;
      const tc = step.trueClass ?? 0;
      const trueLabel = this.s.datasetMeta()?.classLabels?.[tc] ?? `类${tc}`;
      const pTrue = preds[tc];
      const lossVal = step.loss ?? 0;
      // Step 1: show softmax output (compact)
      const probRows = preds.map((p, i) => {
        const label = this.s.datasetMeta()?.classLabels?.[i] ?? `类${i}`;
        return `<div class="fv-term-row${i===tc?' fv-sum-row':''}"><span class="fv-tl">p<sub>${i}</sub></span><span class="fv-op">${label}</span><span class="${i===tc?'fv-af':'fv-r'}">${(p*100).toFixed(1)}%</span></div>`;
      }).join('');
      // Step 2: pick out the true-class probability
      const pickRow = `<div class="fv-term-row fv-bias-row"><span class="fv-tl">p<sub>真实</sub></span><span class="fv-op">真实类别 = ${trueLabel}</span><span class="fv-af">p<sub>${tc}</sub> = ${this.fmt(pTrue)}</span></div>`;
      // Step 3: CE = -ln(p_true), expanded
      const step1 = `<div class="fv-term-row"><span class="fv-tl">CE</span><span class="fv-op">= −ln(p<sub>真实</sub>)</span><span class="fv-st">交叉熵公式</span></div>`;
      const step2 = `<div class="fv-term-row"><span class="fv-tl"></span><span class="fv-op">= −ln(</span><span class="fv-r">${this.fmt(pTrue)}</span><span class="fv-op">)</span></div>`;
      const step3 = `<div class="fv-term-row fv-sum-row"><span class="fv-tl"></span><span class="fv-op">=</span><span class="fv-af">${this.fmt(lossVal)}</span></div>`;
      return `<div class="fv-section-label">损失计算</div><div class="fv-terms">${probRows}</div><div class="fv-sep"></div><div class="fv-terms">${pickRow}${step1}${step2}${step3}</div>`;
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
      const sum = incoming.reduce((s, e) => s + (wBefore[ni]?.[e.neuronFrom] ?? e.weight) * (acts[e.layerFrom]?.[e.neuronFrom] ?? 0), 0) + bBefore;

      // ---- build weighted-sum term rows ----
      const termRows = incoming.map((e, idx) => {
        const a = acts[e.layerFrom]?.[e.neuronFrom] ?? 0;
        const w = wBefore[ni]?.[e.neuronFrom] ?? e.weight;
        const prod = w * a;
        return `<div class="fv-term-row"><span class="fv-tl">w<sub>${idx}</sub>·a<sub>${idx}</sub></span><span class="fv-w">${this.fmt(w)}</span><span class="fv-op">×</span><span class="fv-a">${this.fmt(a)}</span><span class="fv-op">=</span><span class="fv-st">${this.fmt(prod)}</span></div>`;
      });
      const biasRow = `<div class="fv-term-row fv-bias-row"><span class="fv-tl">+ b</span><span class="fv-r">${this.fmt(bBefore)}</span></div>`;
      const sumRow = `<div class="fv-term-row fv-sum-row"><span class="fv-tl">Z</span><span class="fv-r">${this.fmt(sum)}</span></div>`;
      const inner = `<div class="fv-section-label">加权和计算</div><div class="fv-terms">${termRows.join('')}</div><div class="fv-sep"></div>${biasRow}${sumRow}`;
      const actName = layers[li]?.params?.['activation'] ?? '线性';
      const actResult = actName === 'relu' ? Math.max(0, sum) : actName === 'sigmoid' ? 1 / (1 + Math.exp(-sum)) : actName === 'tanh' ? Math.tanh(sum) : sum;
      if (layers[li]?.type === 'output' && actName === 'softmax') {
        const zVals = cache.preActivation![0];
        const softmaxVals = cache.output[0];
        // exp rows: e^z_i = exp(z_i)
        const expRows = zVals.map((z, i) => {
          const expZ = Math.exp(z);
          return `<div class="fv-term-row"><span class="fv-tl">e<sup>z${i}</sup></span><span class="fv-st">exp(${this.fmt(z)})</span><span class="fv-op">=</span><span class="fv-r">${this.fmt(expZ)}</span></div>`;
        }).join('');
        const expSum = zVals.reduce((s, z) => s + Math.exp(z), 0);
        const expSumRow = `<div class="fv-term-row fv-bias-row"><span class="fv-tl">Σ e<sup>Z</sup></span><span class="fv-r">${this.fmt(expSum)}</span></div>`;
        // probability rows: p_i = e^z_i / Σ e^Z
        const probRows = softmaxVals.map((p, i) => {
          return `<div class="fv-term-row fv-sum-row"><span class="fv-tl">p<sub>${i}</sub></span><span class="fv-st">${this.fmt(Math.exp(zVals[i]))}/${this.fmt(expSum)}</span><span class="fv-op">=</span><span class="${i===ni?'fv-af':'fv-r'}">${this.fmt(p)}</span></div>`;
        }).join('');
        const softmaxBlock = `<div class="fv-sep"></div><div class="fv-section-label">softmax 归一化</div><div class="fv-terms">${expRows}${expSumRow}<div class="fv-sep"></div>${probRows}</div>`;
        return `${inner}${softmaxBlock}`;
      }
      return `${inner}<div class="fv-ar"><span class="ar-op">${actName} →</span> <span class="ar-val">${this.fmt(actResult)}</span></div>`;
    }

    if (ss.type === 'backward') {
      const li = ss.layerPair + 1;
      const grad = step.layerGradients.find(g => g.layerId === layers[li]?.id);
      if (!grad?.weightGradients) return null;
      const cache = step.forwardCache?.find(c => c.layerIndex === li);
      if (!cache?.input?.[0]) return null;
      const aPrev = cache.input[0];
      const isOutput = layers[li]?.type === 'output';
      const act = layers[li]?.params?.['activation'] ?? 'relu';
      const counts = this.counts();
      const prevCount = counts[li - 1] ?? 0;
      const thisCount = counts[li] ?? 0;

      // ---- 1) dZ: loss gradient w.r.t. Z ----
      const dZ = grad.biasGradients ?? new Array(thisCount).fill(0);
      let dZRows = '';
      if (isOutput && act === 'softmax') {
        const preds = step.predictions ?? [];
        const tc = step.trueClass ?? 0;
        dZRows = dZ.map((dzVal, i) => {
          const p = preds[i] ?? 0;
          const target = i === tc ? '1' : '0';
          return `<div class="fv-term-row"><span class="fv-tl">dZ<sub>${i}</sub></span><span class="fv-op">= softmax<sub>${i}</sub> − 𝟙</span><span class="fv-st">(label=${tc}) = ${this.fmt(p)}−${target}</span><span class="fv-op">=</span><span class="fv-w">${this.fmt(dzVal)}</span></div>`;
        }).join('');
      } else {
        const dA = grad.inputGradient?.[0] ?? new Array(thisCount).fill(0);
        const Z = cache.preActivation?.[0] ?? new Array(thisCount).fill(0);
        const A = cache.output?.[0] ?? new Array(thisCount).fill(0);
        dZRows = dZ.map((dzVal, i) => {
          let deriv = '';
          if (act === 'relu') deriv = `ReLU'(<span class="fv-st">${this.fmt(Z[i])}</span>) = <span class="fv-st">${Z[i] > 0 ? '1' : '0'}</span>`;
          else if (act === 'sigmoid') deriv = `σ'(<span class="fv-st">${this.fmt(A[i])}</span>) = <span class="fv-st">${this.fmt(A[i] * (1 - A[i]))}</span>`;
          else if (act === 'tanh') deriv = `tanh'(<span class="fv-st">${this.fmt(A[i])}</span>) = <span class="fv-st">${this.fmt(1 - A[i] * A[i])}</span>`;
          else deriv = '×1';
          return `<div class="fv-term-row"><span class="fv-tl">dZ<sub>${i}</sub></span><span class="fv-op">= dA<sub>${i}</sub> × ${deriv}</span><span class="fv-op">=</span><span class="fv-st">${this.fmt(dA[i] ?? 0)}</span><span class="fv-op">×</span><span class="fv-st">${this.fmt(1)}</span><span class="fv-op">=</span><span class="fv-w">${this.fmt(dzVal)}</span></div>`;
        }).join('');
      }
      const dZBlock = `<div class="fv-section-label">dZ — 损失对 Z 的梯度</div><div class="fv-terms">${dZRows}</div>`;

      // ---- 2) dW = a_prev · dZ (outer product) ----
      const dW = grad.weightGradients ?? [];
      let dWRows = '';
      for (let i = 0; i < Math.min(thisCount, dW.length); i++) {
        for (let j = 0; j < Math.min(prevCount, dW[i]?.length ?? 0); j++) {
          const w = dW[i][j];
          dWRows += `<div class="fv-term-row"><span class="fv-tl">dW<sub>${i},${j}</sub></span><span class="fv-op">= a<sub>${j}</sub> × dZ<sub>${i}</sub> =</span><span class="fv-a">${this.fmt(aPrev[j] ?? 0)}</span><span class="fv-op">×</span><span class="fv-w">${this.fmt(dZ[i] ?? 0)}</span><span class="fv-op">=</span><span class="fv-r">${this.fmt(w)}</span></div>`;
        }
      }
      const dWBlock = `<div class="fv-sep"></div><div class="fv-section-label">dW — 权重梯度 (a_prev · dZ)</div><div class="fv-terms">${dWRows}</div>`;

      // ---- 3) db = dZ ----
      const dbRows = dZ.map((dzVal, i) =>
        `<div class="fv-term-row fv-bias-row"><span class="fv-tl">db<sub>${i}</sub></span><span class="fv-op">= dZ<sub>${i}</sub> =</span><span class="fv-r">${this.fmt(dzVal)}</span></div>`
      ).join('');
      const dbBlock = `<div class="fv-sep"></div><div class="fv-section-label">db — 偏置梯度</div><div class="fv-terms">${dbRows}</div>`;

      return `${dZBlock}${dWBlock}${dbBlock}`;
    }

    if (ss.type === 'update') {
      const li = ss.layerIdx;
      const ni = this.getFormulaNeuron(li);
      const snap = step.parameterSnapshots.find(s => s.layerId === layers[li]?.id);
      const wBefore = snap?.weightsBefore?.[ni]?.[0];
      const wAfter = snap?.weightsAfter?.[ni]?.[0];
      if (wBefore == null || wAfter == null) return null;
      return `<span class="fv-lb">更新:</span> ${layers[li]?.name ?? 'L'+li}<span class="fv-op">·</span>神经元${ni}: W<sub>${ni},0</sub> <span class="fv-op">=</span> <span class="fv-bf">${this.fmt(wBefore)}</span> <span class="fv-op">→</span> <span class="fv-af">${this.fmt(wAfter)}</span> <span class="fv-op">(Δ=</span>${this.fmtV(Math.abs(wAfter-wBefore),'fv-r')}<span class="fv-op">)</span>`;
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
