import { Component, computed, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModeDStateService } from '../../services/mode-d-state.service';

@Component({
  selector: 'app-mode-d-floating-charts',
  imports: [CommonModule],
  templateUrl: './mode-d-floating-charts.component.html',
  styleUrl: './mode-d-floating-charts.component.css',
})
export class ModeDFloatingChartsComponent {
  constructor(readonly s: ModeDStateService) {}

  readonly classColors = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6'];

  readonly datasetPoints = computed(() => {
    return this.s.currentDataset().map((pt, i) => ({
      x: pt.input[0], y: pt.input[1], label: pt.label,
      current: i === this.s.currentSampleIndex(),
    }));
  });

  /** Actual data range with 5% margin */
  readonly dataRange = computed(() => {
    const pts = this.datasetPoints();
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const p of pts) {
      if (p.x < xMin) xMin = p.x;
      if (p.x > xMax) xMax = p.x;
      if (p.y < yMin) yMin = p.y;
      if (p.y > yMax) yMax = p.y;
    }
    const xPad = Math.max((xMax - xMin) * 0.08, 0.02);
    const yPad = Math.max((yMax - yMin) * 0.08, 0.02);
    return { xMin: xMin - xPad, xMax: xMax + xPad, yMin: yMin - yPad, yMax: yMax + yPad };
  });

  /** Map data coords to pixel coords within [left, right] / [top, bottom] */
  mapX(v: number, left: number, right: number): number {
    const r = this.dataRange();
    return left + ((v - r.xMin) / (r.xMax - r.xMin)) * (right - left);
  }
  mapY(v: number, top: number, bottom: number): number {
    const r = this.dataRange();
    return bottom - ((v - r.yMin) / (r.yMax - r.yMin)) * (bottom - top);
  }

  private buildCurveLines(viewW: number, viewH: number) {
    const current = this.s.lossHistory();
    const avgLoss = this.s.avgLossHistory();
    const saved = this.s.savedCurves();
    let maxLoss = 0.1;
    for (const c of saved) {
      for (const p of c.points) { if (p.loss > maxLoss) maxLoss = p.loss; }
    }
    for (const p of current) { if (p.loss > maxLoss) maxLoss = p.loss; }
    for (const p of avgLoss) { if (p.loss > maxLoss) maxLoss = p.loss; }

    const pad = viewW > 200 ? 40 : 0;
    const chartW = viewW - pad * 2;
    const chartH = viewH - 10;

    const toSvg = (pts: { iteration: number; loss: number }[]) => {
      if (pts.length < 2) return '';
      return pts.map((p, i) => {
        const x = pad + (i / Math.max(pts.length - 1, 1)) * chartW;
        const y = chartH - (p.loss / maxLoss) * chartH;
        return `${x},${y}`;
      }).join(' ');
    };

    interface CurveLine { label: string; color: string; points: string; dashed: boolean; faint: boolean; }
    const lines: CurveLine[] = saved.map(c => ({ label: c.label, color: c.color, points: toSvg(c.points), dashed: true, faint: false }));
    // Per-step raw loss (faint)
    if (current.length > 1) {
      lines.push({
        label: '单步损失 (原始)',
        color: '#cbd5e1', points: toSvg(current), dashed: false, faint: true,
      });
    }
    // Average loss over all samples (smooth, prominent)
    if (avgLoss.length > 1) {
      lines.push({
        label: `平均损失 (${this.s.trainingConfig().optimizer}+${this.s.currentActivation()})`,
        color: '#d97706', points: toSvg(avgLoss), dashed: false, faint: false,
      });
    }
    return { lines, maxLoss };
  }

  readonly smallCurves = computed(() => this.buildCurveLines(200, 80));

  // ---- modal ----
  readonly showModal = signal(false);
  readonly showBoundaryModal = signal(false);

  deleteCurve(lineIdx: number): void {
    const lines = this.smallCurves().lines;
    const line = lines[lineIdx];
    if (!line?.dashed) return;
    const saved = this.s.savedCurves();
    const idx = saved.findIndex(c => c.label === line.label);
    if (idx >= 0) this.s.deleteSavedCurve(idx);
  }
  readonly modalCurves = computed(() => this.buildCurveLines(560, 280));

  /** Decision boundary grid cells for scatterplot overlay */
  readonly boundaryCells = computed(() => {
    const b = this.s.decisionBoundary();
    if (!b?.grid) return [];
    const res = b.resolution;
    const left = 20, right = 80, top = 20, bottom = 80;
    const cells: { x: number; y: number; s: number; cls: number }[] = [];
    const dx = (b.xMax - b.xMin) / (res - 1);
    const dy = (b.yMax - b.yMin) / (res - 1);
    const cellW = (right - left) / (res - 1);
    for (let yi = 0; yi < res; yi++) {
      for (let xi = 0; xi < res; xi++) {
        const cls = b.grid[yi]?.[xi];
        if (cls == null) continue;
        const dataX = b.xMin + xi * dx;
        const dataY = b.yMin + yi * dy;
        cells.push({
          x: left + ((dataX - b.xMin) / (b.xMax - b.xMin)) * (right - left),
          y: bottom - ((dataY - b.yMin) / (b.yMax - b.yMin)) * (bottom - top),
          s: cellW, cls,
        });
      }
    }
    return cells;
  });

  readonly bcColors = ['rgba(59,130,246,0.25)', 'rgba(245,158,11,0.25)', 'rgba(16,185,129,0.25)', 'rgba(239,68,68,0.25)'];

  /** Large boundary cells for the modal (50x50 grid) */
  readonly modalBoundaryCells = computed(() => {
    const b = this.s.decisionBoundary();
    if (!b?.grid) return [];
    const res = b.resolution;
    const left = 20, right = 280, top = 20, bottom = 280;
    const cellW = (right - left) / (res - 1);
    const cells: { x: number; y: number; w: number; cls: number }[] = [];
    const dx = (b.xMax - b.xMin) / (res - 1);
    const dy = (b.yMax - b.yMin) / (res - 1);
    for (let yi = 0; yi < res; yi++) {
      for (let xi = 0; xi < res; xi++) {
        const cls = b.grid[yi]?.[xi];
        if (cls == null) continue;
        const dataX = b.xMin + xi * dx;
        const dataY = b.yMin + yi * dy;
        cells.push({
          x: left + ((dataX - b.xMin) / (b.xMax - b.xMin)) * (right - left),
          y: bottom - ((dataY - b.yMin) / (b.yMax - b.yMin)) * (bottom - top),
          w: cellW + 0.5, cls,
        });
      }
    }
    return cells;
  });

  readonly hasStep = computed(() => !!this.s.currentStep());
  readonly accuracy = computed(() => this.s.latestAccuracy());
  readonly accPct = computed(() => (this.accuracy() * 100).toFixed(1));
  readonly curSample = computed(() => this.s.currentDataset()[this.s.currentSampleIndex()] ?? null);
  readonly lossVal = computed(() => this.s.currentStep()?.loss?.toFixed(4) ?? '—');
  readonly predLabel = computed(() => this.s.predictedClassLabel());
  readonly trueLabel = computed(() => this.s.trueClassLabel());
  readonly iteration = computed(() => this.s.currentIteration());

  fmt(v: number): string {
    if (Math.abs(v) < 0.0001) return '0';
    return v.toFixed(3);
  }
}
