import { CommonModule, DecimalPipe } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { Subscription, interval } from 'rxjs';
import { AuthService } from '../../services/auth.service';
import type { AuthUser } from '../../models/auth.models';
import { MODE_E_VOCABULARY, runTransformerBlock } from './mode-e-transformer.engine';
import type { AttentionHeadTrace, Matrix, TransformerPreset, TransformerTrace } from './mode-e.types';

@Component({
  selector: 'app-mode-e-page',
  imports: [CommonModule, FormsModule, RouterLink, DecimalPipe],
  templateUrl: './mode-e-page.component.html',
  styleUrl: './mode-e-page.component.css',
})
export class ModeEPageComponent implements OnInit, OnDestroy {
  readonly presets: TransformerPreset[] = [
    { id: 'basic', label: '上下文关系', text: 'transformer learns context' },
    { id: 'attention', label: '注意力分配', text: 'attention tokens share meaning' },
    { id: 'causal', label: 'GPT Mask', text: 'words build future relation' },
  ];
  readonly vocabulary = MODE_E_VOCABULARY;

  user: AuthUser | null = null;
  text = this.presets[0]!.text;
  selectedPresetId = this.presets[0]!.id;
  causalMask = false;
  selectedHead = 0;
  activeStepIndex = 0;
  playing = true;

  trace: TransformerTrace = runTransformerBlock(this.text, this.causalMask);

  private readonly subs = new Subscription();

  constructor(private readonly authSvc: AuthService) {}

  ngOnInit(): void {
    this.subs.add(this.authSvc.user$.subscribe(user => { this.user = user; }));
    this.authSvc.restoreSession();
    this.subs.add(
      interval(2400).subscribe(() => {
        if (this.playing) {
          this.activeStepIndex = (this.activeStepIndex + 1) % this.trace.steps.length;
        }
      }),
    );
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  logout(): void {
    this.authSvc.logout();
  }

  applyPreset(id: string): void {
    const preset = this.presets.find(item => item.id === id);
    if (!preset) {
      return;
    }
    this.selectedPresetId = id;
    this.text = preset.text;
    this.causalMask = preset.id === 'causal';
    this.recompute();
  }

  recompute(): void {
    this.trace = runTransformerBlock(this.text, this.causalMask);
    if (this.selectedHead >= this.trace.heads.length) {
      this.selectedHead = 0;
    }
    if (this.activeStepIndex >= this.trace.steps.length) {
      this.activeStepIndex = 0;
    }
  }

  selectStep(index: number): void {
    this.activeStepIndex = index;
    this.playing = false;
  }

  nextStep(): void {
    this.activeStepIndex = (this.activeStepIndex + 1) % this.trace.steps.length;
  }

  previousStep(): void {
    this.activeStepIndex = (this.activeStepIndex + this.trace.steps.length - 1) % this.trace.steps.length;
  }

  selectHead(index: number): void {
    this.selectedHead = index;
  }

  diagramActive(...ids: string[]): boolean {
    return ids.includes(this.activeStep.id);
  }

  get activeStep() {
    return this.trace.steps[this.activeStepIndex]!;
  }

  get activeHead(): AttentionHeadTrace {
    return this.trace.heads[this.selectedHead]!;
  }

  get attentionRowSums(): number[] {
    return this.activeHead.weights.map(row => row.reduce((sum, value) => sum + value, 0));
  }

  get maxAbsActiveMatrix(): number {
    return this.maxAbs(this.activeStep.matrix);
  }

  get maxAbsBlockOutput(): number {
    return this.maxAbs(this.trace.blockOutput);
  }

  cellBackground(value: number, maxAbs: number): string {
    if (!Number.isFinite(value)) {
      return '#f1f5f9';
    }
    const intensity = maxAbs === 0 ? 0 : Math.min(Math.abs(value) / maxAbs, 1);
    if (value >= 0) {
      return `rgba(37, 99, 235, ${0.08 + intensity * 0.36})`;
    }
    return `rgba(217, 119, 6, ${0.08 + intensity * 0.34})`;
  }

  attentionBackground(value: number): string {
    const alpha = 0.08 + Math.min(Math.max(value, 0), 1) * 0.72;
    return `rgba(5, 150, 105, ${alpha})`;
  }

  formatNumber(value: number): string {
    if (!Number.isFinite(value)) {
      return '-inf';
    }
    const abs = Math.abs(value);
    if (abs < 0.0001) {
      return '0.0000';
    }
    return value.toFixed(4);
  }

  formatRow(row: number[]): string {
    return row.map(value => this.formatNumber(value)).join('  ');
  }

  rowLabel(index: number): string {
    return this.trace.tokens[index] ?? `t${index}`;
  }

  colLabel(index: number): string {
    return this.trace.tokens[index] ?? `c${index}`;
  }

  matrixColumns(matrix: Matrix): number[] {
    return Array.from({ length: matrix[0]?.length ?? 0 }, (_, index) => index);
  }

  matrixRows(matrix: Matrix): number[] {
    return Array.from({ length: matrix.length }, (_, index) => index);
  }

  private maxAbs(matrix: Matrix): number {
    const values = matrix.flat().filter(Number.isFinite).map(Math.abs);
    return values.length ? Math.max(...values, 1e-9) : 1;
  }
}
