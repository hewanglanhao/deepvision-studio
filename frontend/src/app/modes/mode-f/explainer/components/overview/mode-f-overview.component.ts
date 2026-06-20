import { Component, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TeachingTermDirective } from '@shared/teaching/teaching-term.directive';
import { ModeFStateService } from '../../services/mode-f-state.service';

const CELL_W = 130, CELL_H = 80, GAP = 50, PAD = 50;
// topology layout
const TLG = 160, TNG = 52, TNR = 18, TPAD = 50;

@Component({
  selector: 'app-mode-f-overview',
  imports: [CommonModule, TeachingTermDirective],
  templateUrl: './mode-f-overview.component.html',
  styleUrl: './mode-f-overview.component.css',
})
export class ModeFOverviewComponent {
  readonly CELL_W = CELL_W; readonly CELL_H = CELL_H;
  readonly TNR = TNR; readonly TPAD = TPAD;
  constructor(readonly s: ModeFStateService) {}

  // ---- time-unrolled view ----
  readonly step = computed(() => this.s.currentStep());
  readonly loss = computed(() => this.s.currentStep()?.loss.toFixed(4) ?? '—');
  readonly iter = computed(() => this.s.currentIteration());
  readonly accuracy = computed(() => (this.s.latestAccuracy() * 100).toFixed(1));
  readonly meta = computed(() => this.s.networkMeta());
  readonly lossHistory = computed(() => this.s.lossHistory());
  readonly avgLossHistory = computed(() => this.s.avgLossHistory());

  readonly curSampleInputs = computed(() => {
    const ds = this.s.currentDataset();
    const idx = this.s.currentSampleIndex();
    return ds[idx]?.inputs ?? [];
  });

  readonly timeSteps = computed(() => this.step()?.forwardResult?.states?.length ?? 0);
  readonly hiddenDim = computed(() => this.step()?.hiddenDim ?? 0);
  readonly svgW = computed(() => Math.max(this.timeSteps() * (CELL_W + GAP) + PAD * 2, 600));
  readonly svgH = computed(() => this.hiddenDim() * 22 + CELL_H + PAD * 2);

  readonly rawLossPoints = computed(() => {
    const raw = this.lossHistory(), avg = this.avgLossHistory();
    if (raw.length < 2) return '';
    const max = Math.max(...raw.map(p => p.loss), ...avg.map(p => p.loss), 0.1);
    return raw.map((p, i) => { const x = (i / Math.max(raw.length - 1, 1)) * 200; const y = 60 - (p.loss / max) * 60; return `${x},${y}`; }).join(' ');
  });
  readonly avgLossPoints = computed(() => {
    const pts = this.avgLossHistory();
    if (pts.length < 2) return '';
    const all = this.lossHistory();
    const max = Math.max(...all.map(p => p.loss), ...pts.map(p => p.loss), 0.1);
    return pts.map((p, i) => { const x = (i / Math.max(pts.length - 1, 1)) * 200; const y = 60 - (p.loss / max) * 60; return `${x},${y}`; }).join(' ');
  });

  cellX(t: number): number { return PAD + t * (CELL_W + GAP); }
  cellY(): number { return PAD + 30; }

  barH(val: number): number { return Math.max(2, Math.abs(val) * CELL_H * 0.35); }
  maxBarH(arr: number[]): number {
    let m = 2;
    for (const v of arr) { const h = Math.abs(v) * CELL_H * 0.35; if (h > m) m = h; }
    return m;
  }
  barColor(v: number): string { return v > 0 ? '#2563eb' : v < 0 ? '#dc2626' : '#cbd5e1'; }

  // ---- topology graph ----
  readonly topoCounts = computed(() => this.s.neuronCounts());
  readonly topoEdges = computed(() => this.s.weightEdges());
  readonly topoBiases = computed(() => this.s.biasValues());
  readonly topoActs = computed(() => this.s.neuronActivations());
  readonly topoMaxN = computed(() => Math.max(...this.topoCounts(), 2));
  readonly topoLayerNames = ['输入层', '隐层 (tanh)', '输出层 (softmax)'];
  readonly topoSvgW = computed(() => 2 * TLG + TPAD * 2 + 60);
  readonly topoSvgH = computed(() => this.topoMaxN() * TNG + TPAD * 2);

  // Hidden layer box (rounded rect enclosing hidden neurons)
  readonly hiddenBox = computed(() => {
    const n = this.topoCounts()[1];
    const x = this.topoNx(1) - TNR - 12;
    const y = this.topoNy(0, 1) - TNR - 8;
    const w = TNR * 2 + 24;
    const h = n > 1 ? this.topoNy(n - 1, 1) - this.topoNy(0, 1) + TNR * 2 + 8 : TNR * 2 + 16;
    return { x, y, w, h };
  });

  // Self-loop arrow: from right side of hidden box, loop up, back to top of box
  readonly selfLoopPath = computed(() => {
    const b = this.hiddenBox();
    const rx = b.x + b.w;
    const ry = b.y;
    const r = 24;
    return `M ${rx} ${b.y + b.h * 0.3} C ${rx + r} ${b.y + b.h * 0.3}, ${rx + r} ${ry - r + 4}, ${b.x + b.w / 2} ${ry - 4}`;
  });

  readonly hoveredTopoEdge = signal<{ layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number } | null>(null);
  labelX = 0; labelY = 0;

  onTopoSvgMove(e: MouseEvent): void {
    const svg = e.currentTarget as SVGSVGElement;
    if (!svg) return;
    const r = svg.getBoundingClientRect();
    this.labelX = e.clientX - r.left;
    this.labelY = e.clientY - r.top;
  }

  topoNx(li: number): number { return TPAD + 30 + li * TLG; }
  topoNy(ni: number, li: number): number {
    const cnt = this.topoCounts()[li] ?? 2;
    return TPAD + (this.topoMaxN() - cnt) * TNG / 2 + ni * TNG + TNG / 2;
  }

  topoEdgeStroke(e: { type: string }): string {
    return e.type === 'wxh' ? '#3b82f6' : '#10b981';
  }

  topoEdgeOpacity(e: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number; type: string }): number {
    const h = this.hoveredTopoEdge();
    if (h) {
      if (h.layerFrom === e.layerFrom && h.layerTo === e.layerTo && h.neuronFrom === e.neuronFrom && h.neuronTo === e.neuronTo) return 1;
      return h.layerFrom === e.layerFrom && h.layerTo === e.layerTo ? 0.20 : 0.12;
    }
    return 1;
  }

  topoEdgeWidth(e: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number }): number {
    const h = this.hoveredTopoEdge();
    if (h && h.layerFrom === e.layerFrom && h.layerTo === e.layerTo && h.neuronFrom === e.neuronFrom && h.neuronTo === e.neuronTo) return 2.5;
    return 1;
  }

  topoEdgeLabelPos(e: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number }): { mx: number; my: number; angle: number } {
    const x1 = this.topoNx(e.layerFrom), y1 = this.topoNy(e.neuronFrom, e.layerFrom);
    const x2 = this.topoNx(e.layerTo), y2 = this.topoNy(e.neuronTo, e.layerTo);
    const dx = x2 - x1, dy = y2 - y1;
    const len = Math.sqrt(dx * dx + dy * dy) || 1;
    const nx = -dy / len, ny = dx / len;  // unit normal
    const mx = (x1 + x2) / 2 - nx * 8;
    const my = (y1 + y2) / 2 - ny * 8;
    const angle = Math.atan2(dy, dx) * 180 / Math.PI;
    return { mx, my, angle };
  }

  topoNfill(li: number, ni: number): string {
    const val = this.topoActs()[li]?.[ni] ?? 0;
    const abs = Math.abs(val);
    const intensity = Math.min(abs / 2, 1);
    if (abs < 0.001) return 'rgba(148,163,184,0.4)';
    if (val > 0) return `rgba(37,99,235,${0.4 + intensity * 0.5})`;
    return `rgba(220,38,38,${0.3 + intensity * 0.5})`;
  }

  readonly selTopoRef = computed(() => this.s.selectedNeuronRef());
  topoIsSelected(li: number, ni: number): boolean {
    const r = this.selTopoRef();
    return r?.layerIdx === li && r?.neuronIdx === ni;
  }

  selectTopoN(li: number, ni: number): void { this.s.selectNeuron(li, ni); }

  // ---- hover tooltip ----
  readonly hoveredEdgeLabel = computed(() => {
    const h = this.hoveredTopoEdge();
    if (!h) return null;
    const match = this.topoEdges().find(e =>
      e.layerFrom === h.layerFrom && e.neuronFrom === h.neuronFrom &&
      e.layerTo === h.layerTo && e.neuronTo === h.neuronTo
    );
    if (!match) return null;
    return `w=${this.fmt(match.weight)}`;
  });

  onTopoEdgeEnter(e: { layerFrom: number; neuronFrom: number; layerTo: number; neuronTo: number }): void {
    this.hoveredTopoEdge.set(e);
  }
  onTopoEdgeLeave(): void { this.hoveredTopoEdge.set(null); }

  // ---- shared ----
  fmt(v: number): string {
    if (Math.abs(v) < 0.0001) return '0';
    if (Math.abs(v) < 0.01) return v.toFixed(4);
    return v.toFixed(3);
  }
  fmtShort(v: number): string {
    if (Math.abs(v) < 0.005) return '0';
    return v.toFixed(2);
  }
  max(a: number, b: number): number { return Math.max(a, b); }
  fmtProbs(probs: number[]): string { return probs.map(p => (p * 100).toFixed(0) + '%').join(' / '); }
  fmtOutput(output: { output: number[] }): string { return this.fmtProbs(output.output); }
}
