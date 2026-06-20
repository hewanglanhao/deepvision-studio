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

  readonly neuron = computed(() => this.state.selectedNeuron());
  readonly layerNames = ['输入层', '隐层 (tanh)', '输出层 (softmax)'];

  sourceLayerName(layerIdx: number): string { return this.layerNames[layerIdx] ?? `L${layerIdx}`; }

  readonly gradientNorm = computed(() => this.step()?.gradient?.gradientNorm);
  readonly hiddenStates = computed(() => this.step()?.forwardResult?.states ?? []);
  readonly timeSteps = computed(() => this.step()?.timeSteps ?? 0);
  readonly hiddenDim = computed(() => this.step()?.hiddenDim ?? 0);
  readonly outputProbs = computed(() => this.step()?.outputProbs ?? []);

  readonly weightMatrices = computed(() => {
    const step = this.step();
    const engine = this.state.engine;
    if (!engine) return [];
    const snap = step?.weightSnapshot;
    const Wxh = snap?.WxhAfter ?? engine.Wxh ?? [];
    const Whh = snap?.WhhAfter ?? engine.Whh ?? [];
    const Why = snap?.WhyAfter ?? engine.Why ?? [];
    const bh  = snap?.bhAfter ?? engine.bh ?? [];
    const by_ = snap?.byAfter ?? engine.by ?? [];
    const items: { name: string; label: string; rows: number; cols: number; data: number[][] }[] = [];
    if (Wxh.length) items.push({ name: 'W_xh', label: '输入→隐层', rows: Wxh.length, cols: Wxh[0]?.length ?? 0, data: Wxh });
    if (Whh.length) items.push({ name: 'W_hh', label: '隐层自循环', rows: Whh.length, cols: Whh[0]?.length ?? 0, data: Whh });
    if (Why.length) items.push({ name: 'W_hy', label: '隐层→输出', rows: Why.length, cols: Why[0]?.length ?? 0, data: Why });
    if (bh.length)  items.push({ name: 'b_h', label: '隐层偏置', rows: 1, cols: bh.length, data: [bh] });
    if (by_.length) items.push({ name: 'b_y', label: '输出偏置', rows: 1, cols: by_.length, data: [by_] });
    return items;
  });

  cellColor(v: number): string {
    const a = Math.min(Math.abs(v) / 2, 1);
    if (v >= 0) return `rgba(59,130,246,${0.15 + a * 0.55})`;
    return `rgba(239,68,68,${0.15 + a * 0.45})`;
  }

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
