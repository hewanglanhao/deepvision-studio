import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, OnInit, Output } from '@angular/core';
import { RouterLink } from '@angular/router';
import { AuthUser } from '../../../../models/auth.models';
import { ModeCArticleComponent } from '../article/mode-c-article.component';
import { ModeCDetailPanelComponent } from '../detail-panels/mode-c-detail-panel.component';
import { ModeCOverviewComponent } from '../overview/mode-c-overview.component';
import { ModeCModelService } from '../../services/mode-c-model.service';

@Component({
  selector: 'app-mode-c-explainer-shell',
  imports: [
    CommonModule,
    RouterLink,
    ModeCOverviewComponent,
    ModeCDetailPanelComponent,
    ModeCArticleComponent
  ],
  templateUrl: './mode-c-explainer-shell.component.html',
  styleUrl: './mode-c-explainer-shell.component.css'
})
export class ModeCExplainerShellComponent implements OnInit {
  @Input() user: AuthUser | null = null;
  @Output() readonly logoutRequested = new EventEmitter<void>();

  constructor(readonly model: ModeCModelService) {}

  ngOnInit(): void {
    this.model.initializeNativeShell();
  }

  get isLoggedIn(): boolean {
    return !!this.user;
  }

  requestLogout(): void {
    this.logoutRequested.emit();
  }
}
