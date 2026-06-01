import { Component, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TeachingTermDirective } from '@shared/teaching/teaching-term.directive';
import { ModeFStateService } from '../../services/mode-f-state.service';
import type { ModeFTrainingConfig } from '../../models/mode-f.types';

@Component({
  selector: 'app-mode-f-control-panel',
  imports: [CommonModule, TeachingTermDirective],
  templateUrl: './mode-f-control-panel.component.html',
  styleUrl: './mode-f-control-panel.component.css',
})
export class ModeFControlPanelComponent {
  constructor(readonly state: ModeFStateService) {}

  readonly presets = computed(() => this.state.presetOptions());
  readonly presetId = computed(() => this.state.selectedPresetId());
  readonly config = computed(() => this.state.trainingConfig());
  readonly isPlaying = computed(() => this.state.isPlaying());
  readonly isRunning = computed(() => this.state.isPlaying() || this.state.status() === 'running');
  readonly iteration = computed(() => this.state.currentIteration());
  readonly status = computed(() => this.state.status());

  readonly optimizerOptions: { value: ModeFTrainingConfig['optimizer']; label: string }[] = [
    { value: 'sgd', label: 'SGD' },
    { value: 'momentum', label: 'Momentum' },
    { value: 'adam', label: 'Adam' },
  ];

  readonly speedOptions: { value: number; label: string }[] = [
    { value: 500, label: '慢速' },
    { value: 200, label: '正常' },
    { value: 50, label: '快速' },
  ];

  readonly playSpeed = computed(() => this.state.playSpeed());

  selectPreset(id: string): void {
    this.state.setPreset(id);
  }

  setLr(value: string): void {
    this.state.setTrainingConfig({ learningRate: parseFloat(value) });
  }

  setOptimizer(value: ModeFTrainingConfig['optimizer']): void {
    this.state.setTrainingConfig({ optimizer: value });
  }

  setSpeed(ms: number): void {
    this.state.setPlaySpeed(ms);
  }

  setSteps(val: string): void {
    const n = parseInt(val, 10);
    if (n > 0) this.state.setTrainingConfig({ maxIterations: n });
  }

  stepOnce(): void {
    this.state.stepForward();
  }

  togglePlay(): void {
    this.state.togglePlay();
  }

  reset(): void {
    this.state.reset();
  }
}
