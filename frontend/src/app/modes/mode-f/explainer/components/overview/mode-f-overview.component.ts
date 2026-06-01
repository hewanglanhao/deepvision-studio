import { Component, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModeFStateService } from '../../services/mode-f-state.service';

const CELL_W = 130, CELL_H = 80, GAP = 50, PAD = 50;

@Component({
  selector: 'app-mode-f-overview',
  imports: [CommonModule],
  templateUrl: './mode-f-overview.component.html',
  styleUrl: './mode-f-overview.component.css',
})
export class ModeFOverviewComponent {
  readonly CELL_W = CELL_W; readonly CELL_H = CELL_H;
  constructor(readonly s: ModeFStateService) {}

  readonly step = computed(() => this.s.currentStep());
  readonly loss = computed(() => this.s.currentStep()?.loss.toFixed(4) ?? '—');
  readonly iter = computed(() => this.s.currentIteration());
  readonly accuracy = computed(() => (this.s.latestAccuracy() * 100).toFixed(1));
  readonly meta = computed(() => this.s.networkMeta());
  readonly lossHistory = computed(() => this.s.lossHistory());
  readonly avgLossHistory = computed(() => this.s.avgLossHistory());

  readonly timeSteps = computed(() => this.step()?.forwardResult?.states?.length ?? 0);
  readonly hiddenDim = computed(() => this.step()?.hiddenDim ?? 0);
  readonly svgW = computed(() => Math.max(this.timeSteps() * (CELL_W + GAP) + PAD * 2, 600));
  readonly svgH = computed(() => this.hiddenDim() * 22 + CELL_H + PAD * 2);

  // Loss curve
  readonly lossPoints = computed(() => { const pts = this.lossHistory(); if (pts.length < 2) return ''; const max = Math.max(...pts.map(p => p.loss), 0.1); return pts.map((p, i) => { const x = (i / Math.max(pts.length - 1, 1)) * 200; const y = 60 - (p.loss / max) * 60; return `${x},${y}`; }).join(' '); });

  cellX(t: number): number { return PAD + t * (CELL_W + GAP); }
  cellY(): number { return PAD + 30; }
  barX(i: number): number { return 8 + i * (CELL_W - 24) / this.hiddenDim(); }

  barH(val: number): number { return Math.max(2, Math.abs(val) * CELL_H * 0.35); }
  barY(val: number): number { return this.cellY() + CELL_H / 2 - (val > 0 ? this.barH(val) : 0); }
  barColor(v: number): string { return v > 0 ? '#2563eb' : v < 0 ? '#dc2626' : '#cbd5e1'; }

  fmt(v: number): string { return Math.abs(v) < 0.01 ? '0' : v.toFixed(2); }
  max(a: number, b: number): number { return Math.max(a, b); }
  fmtProbs(probs: number[]): string { return probs.map(p => (p * 100).toFixed(0) + '%').join(' / '); }
  fmtOutput(output: { output: number[] }): string { return this.fmtProbs(output.output); }
}
