import { Component, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { TeachingTermDirective } from '@shared/teaching/teaching-term.directive';
import { ModeDStateService } from '../../services/mode-d-state.service';
import type { ModeDOptimizer } from '../../models/mode-d.types';

@Component({
  selector: 'app-mode-d-control-panel',
  imports: [CommonModule, TeachingTermDirective],
  templateUrl: './mode-d-control-panel.component.html',
  styleUrl: './mode-d-control-panel.component.css',
})
export class ModeDControlPanelComponent {
  constructor(readonly state: ModeDStateService) {}

  readonly presets = computed(() => this.state.presetOptions());
  readonly presetId = computed(() => this.state.selectedPresetId());
  readonly config = computed(() => this.state.trainingConfig());
  readonly isPlaying = computed(() => this.state.isPlaying());
  readonly iteration = computed(() => this.state.currentIteration());
  readonly status = computed(() => this.state.status());

  readonly optimizerOptions: { value: ModeDOptimizer; label: string }[] = [
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

  setOptimizer(value: ModeDOptimizer): void {
    this.state.setTrainingConfig({ optimizer: value });
  }

  setSpeed(ms: number): void {
    this.state.setPlaySpeed(ms);
  }

  setSteps(val: string): void {
    const n = parseInt(val, 10);
    if (n > 0) this.state.setTrainingConfig({ maxIterations: n });
  }

  readonly isAnimating = computed(() => this.state.isAnimating());
  readonly hasMore = computed(() => this.state.hasMoreSubSteps());
  readonly totalPending = computed(() => this.state.totalPendingSubSteps());
  readonly remaining = computed(() => this.state.remainingSubSteps());

  stepOnce(): void {
    this.state.startAnimatedStep();
  }

  nextSubStep(): void {
    this.state.advanceSubStep();
  }

  togglePlay(): void {
    this.state.togglePlay();
  }

  reset(): void {
    this.state.reset();
  }

  saveCurve(): void {
    this.state.saveCurrentCurve();
  }

  clearCurves(): void {
    this.state.clearSavedCurves();
  }
}
