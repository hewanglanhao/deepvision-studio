import { Component, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModeEStateService } from '../../services/mode-e-state.service';

@Component({
  selector: 'app-mode-e-floating-charts',
  imports: [CommonModule],
  templateUrl: './mode-e-floating-charts.component.html',
  styleUrl: './mode-e-floating-charts.component.css',
})
export class ModeEFloatingChartsComponent {
  constructor(readonly s: ModeEStateService) {}

  readonly classColors = ['#6366f1', '#f59e0b', '#10b981', '#ef4444', '#8b5cf6'];

  readonly datasetPoints = computed(() => {
    return this.s.currentDataset().map((pt, i) => ({
      x: pt.input[0], y: pt.input[1], label: pt.label,
      current: i === this.s.currentSampleIndex(),
    }));
  });

  readonly lossSvgPoints = computed(() => {
    const pts = this.s.lossHistory();
    if (pts.length < 2) return '';
    const max = Math.max(...pts.map(p => p.loss), 0.1);
    return pts.map((p, i) => {
      const x = (i / Math.max(pts.length - 1, 1)) * 200;
      const y = 80 - (p.loss / max) * 80;
      return `${x},${y}`;
    }).join(' ');
  });

  readonly hasStep = computed(() => !!this.s.currentStep());
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
