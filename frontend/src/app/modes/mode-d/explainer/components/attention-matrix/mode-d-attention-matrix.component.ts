import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { ModeDStateService } from '../../services/mode-d-state.service';

@Component({
  selector: 'app-mode-d-attention-matrix',
  imports: [CommonModule],
  templateUrl: './mode-d-attention-matrix.component.html',
  styleUrl: './mode-d-attention-matrix.component.css'
})
export class ModeDAttentionMatrixComponent {
  constructor(readonly state: ModeDStateService) {}

  cellOpacity(value: number): number {
    return 0.12 + value * 0.88;
  }

  formatWeight(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  isFocusedCell(row: number, col: number): boolean {
    const focus = this.state.activeAttentionDetail();
    return focus.row === row && focus.col === col;
  }

  isDimmedCell(row: number, col: number): boolean {
    const selected = this.state.selectedCell();
    if (!selected) {
      return false;
    }
    return selected.row !== row || selected.col !== col;
  }

  isHighlightedRow(row: number): boolean {
    return this.state.activeAttentionDetail().row === row;
  }

  isHighlightedCol(col: number): boolean {
    return this.state.activeAttentionDetail().col === col;
  }
}

