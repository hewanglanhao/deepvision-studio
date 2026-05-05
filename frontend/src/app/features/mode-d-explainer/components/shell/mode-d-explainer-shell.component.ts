import { Component, Input, Output, EventEmitter, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import type { AuthUser } from '../../../../models/auth.models';
import { ModeDOverviewComponent } from '../overview/mode-d-overview.component';
import { ModeDDetailPanelComponent } from '../detail-panel/mode-d-detail-panel.component';
import { ModeDControlPanelComponent } from '../control-panel/mode-d-control-panel.component';
import { ModeDFloatingChartsComponent } from '../floating-charts/mode-d-floating-charts.component';
import { ModeDStateService } from '../../services/mode-d-state.service';

@Component({
  selector: 'app-mode-d-explainer-shell',
  imports: [
    CommonModule,
    RouterLink,
    ModeDOverviewComponent,
    ModeDDetailPanelComponent,
    ModeDControlPanelComponent,
    ModeDFloatingChartsComponent,
  ],
  templateUrl: './mode-d-explainer-shell.component.html',
  styleUrl: './mode-d-explainer-shell.component.css',
})
export class ModeDExplainerShellComponent implements OnInit, OnDestroy {
  @Input() user: AuthUser | null = null;
  @Output() readonly logoutRequested = new EventEmitter<void>();

  constructor(readonly state: ModeDStateService) {}

  ngOnInit(): void {
    this.state.loadPreset(this.state.selectedPresetId());
  }

  ngOnDestroy(): void {
    this.state.pause();
  }

  get isLoggedIn(): boolean {
    return !!this.user;
  }

  requestLogout(): void {
    this.logoutRequested.emit();
  }
}
