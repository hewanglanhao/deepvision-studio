import { Component, Input, Output, EventEmitter, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterLink } from '@angular/router';
import type { AuthUser } from '@core/auth/auth.models';
import { ModeEOverviewComponent } from '../overview/mode-e-overview.component';
import { ModeEDetailPanelComponent } from '../detail-panel/mode-e-detail-panel.component';
import { ModeEControlPanelComponent } from '../control-panel/mode-e-control-panel.component';
import { ModeEFloatingChartsComponent } from '../floating-charts/mode-e-floating-charts.component';
import { ModeEStateService } from '../../services/mode-e-state.service';

@Component({
  selector: 'app-mode-e-explainer-shell',
  imports: [
    CommonModule,
    RouterLink,
    ModeEOverviewComponent,
    ModeEDetailPanelComponent,
    ModeEControlPanelComponent,
    ModeEFloatingChartsComponent,
  ],
  templateUrl: './mode-e-explainer-shell.component.html',
  styleUrl: './mode-e-explainer-shell.component.css',
})
export class ModeEExplainerShellComponent implements OnInit, OnDestroy {
  @Input() user: AuthUser | null = null;
  @Output() readonly logoutRequested = new EventEmitter<void>();

  constructor(readonly state: ModeEStateService) {}

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
