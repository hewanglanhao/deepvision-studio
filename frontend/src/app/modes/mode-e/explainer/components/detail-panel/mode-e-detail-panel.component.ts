import { Component, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModeEStateService } from '../../services/mode-e-state.service';

@Component({
  selector: 'app-mode-e-detail-panel',
  imports: [CommonModule],
  templateUrl: './mode-e-detail-panel.component.html',
  styleUrl: './mode-e-detail-panel.component.css',
})
export class ModeEDetailPanelComponent {
  constructor(readonly state: ModeEStateService) {}

  readonly neuron = computed(() => this.state.selectedNeuron());
  readonly phase = computed(() => this.state.activePhase());

  abs(v: number): number { return Math.abs(v); }

  fmt(v: number): string {
    if (Math.abs(v) < 0.0001) return '0';
    if (Math.abs(v) < 0.01) return v.toFixed(5);
    return v.toFixed(4);
  }

  weightColor(v: number): string {
    const abs = Math.abs(v);
    const a = Math.min(abs / 3, 1);
    if (v >= 0) return `rgba(59,130,246,${0.15 + a * 0.7})`;
    return `rgba(239,68,68,${0.15 + a * 0.5})`;
  }

  gradientColor(v: number): string {
    const a = Math.min(Math.abs(v) / 2, 1);
    if (v >= 0) return `rgba(16,185,129,${0.15 + a * 0.7})`;
    return `rgba(245,158,11,${0.15 + a * 0.7})`;
  }

  changeColor(v: number): string {
    const a = Math.min(Math.abs(v) * 5, 1);
    return `rgba(167,139,250,${0.15 + a * 0.7})`;
  }

  // get the name of the source layer for an incoming edge
  sourceLayerName(edge: { layerFrom: number }): string {
    const layers = this.state.networkLayers();
    return layers[edge.layerFrom]?.name ?? `L${edge.layerFrom}`;
  }

  readonly classes = computed(() => this.state.datasetMeta()?.classLabels ?? []);
}
