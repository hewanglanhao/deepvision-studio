import { Component, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TeachingTermDirective } from '@shared/teaching/teaching-term.directive';
import { ModeFStateService } from '../../services/mode-f-state.service';

@Component({
  selector: 'app-mode-f-detail-panel',
  imports: [CommonModule, TeachingTermDirective],
  templateUrl: './mode-f-detail-panel.component.html',
  styleUrl: './mode-f-detail-panel.component.css',
})
export class ModeFDetailPanelComponent {
  constructor(readonly state: ModeFStateService) {}

  readonly step = computed(() => this.state.currentStep());
  readonly meta = computed(() => this.state.networkMeta());
  readonly datasetMeta = computed(() => this.state.datasetMeta());

  readonly gradientNorm = computed(() => this.step()?.gradient?.gradientNorm);
  readonly hiddenStates = computed(() => this.step()?.forwardResult?.states ?? []);
  readonly timeSteps = computed(() => this.step()?.timeSteps ?? 0);
  readonly hiddenDim = computed(() => this.step()?.hiddenDim ?? 0);
  readonly outputProbs = computed(() => this.step()?.outputProbs ?? []);

  readonly weightShapes = computed(() => {
    const m = this.meta();
    if (!m) return [];
    return [
      { name: 'W_xh (input→hidden)', shape: `${m.hiddenDim}×${m.inputDim}`, desc: '输入到隐层的权重矩阵' },
      { name: 'W_hh (hidden→hidden)', shape: `${m.hiddenDim}×${m.hiddenDim}`, desc: '隐层循环权重矩阵' },
      { name: 'W_hy (hidden→output)', shape: `${m.outputDim}×${m.hiddenDim}`, desc: '隐层到输出的权重矩阵' },
      { name: 'b_h (hidden bias)', shape: `${m.hiddenDim}`, desc: '隐层偏置向量' },
      { name: 'b_y (output bias)', shape: `${m.outputDim}`, desc: '输出层偏置向量' },
    ];
  });

  readonly classLabels = computed(() => this.datasetMeta()?.classLabels ?? []);

  fmt(v: number): string {
    if (Math.abs(v) < 0.0001) return '0';
    if (Math.abs(v) < 0.01) return v.toFixed(5);
    return v.toFixed(4);
  }

  barColor(v: number): string {
    const abs = Math.abs(v);
    const a = Math.min(abs / 1.5, 1);
    if (v >= 0) return `rgba(59,130,246,${0.2 + a * 0.6})`;
    return `rgba(239,68,68,${0.2 + a * 0.5})`;
  }

  barWidth(v: number): number {
    return Math.max(3, Math.abs(v) * 80);
  }

  gradColor(v: number): string {
    const a = Math.min(v / 2, 1);
    return `rgba(245,158,11,${0.15 + a * 0.7})`;
  }

  abs(v: number): number { return Math.abs(v); }
  min(a: number, b: number): number { return Math.min(a, b); }
  maxArray(arr: number[]): number { return Math.max(...arr); }
}
