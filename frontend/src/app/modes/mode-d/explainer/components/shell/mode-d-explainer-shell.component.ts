import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { AuthUser } from '@core/auth/auth.models';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import { ModeDAttentionMatrixComponent } from '../attention-matrix/mode-d-attention-matrix.component';
import { ModeDInputPanelComponent } from '../input-panel/mode-d-input-panel.component';
import { ModeDQkvPanelComponent } from '../qkv-panel/mode-d-qkv-panel.component';
import { ModeDReportPanelComponent } from '../report-panel/mode-d-report-panel.component';
import { ModeDTopKPanelComponent } from '../topk-panel/mode-d-topk-panel.component';

@Component({
  selector: 'app-mode-d-explainer-shell',
  imports: [
    CommonModule,
    PlatformTopbarComponent,
    ModeDInputPanelComponent,
    ModeDTopKPanelComponent,
    ModeDAttentionMatrixComponent,
    ModeDQkvPanelComponent,
    ModeDReportPanelComponent
  ],
  templateUrl: './mode-d-explainer-shell.component.html',
  styleUrl: './mode-d-explainer-shell.component.css'
})
export class ModeDExplainerShellComponent {
  @Input() user: AuthUser | null = null;
  @Output() readonly logoutRequested = new EventEmitter<void>();

  readonly statusPills = ['Transformer 可解释', '推理数据源可验证'];

  requestLogout(): void {
    this.logoutRequested.emit();
  }
}
