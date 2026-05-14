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
        label: `平均损失 (${this.s.trainingConfig().optimizer.toUpperCase()})`,
        color: '#d97706', points: toSvg(avgLoss), dashed: false, faint: false,
      });
    }
    return { lines, maxLoss };
  }

  readonly smallCurves = computed(() => this.buildCurveLines(200, 80));

  // ---- modal ----
  readonly showModal = signal(false);
  readonly modalCurves = computed(() => this.buildCurveLines(560, 280));

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
