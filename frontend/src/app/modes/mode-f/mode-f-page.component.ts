import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { AuthService } from '@core/auth/auth.service';
import { ModeFExplainerShellComponent } from '@modes/mode-f/explainer/components/shell/mode-f-explainer-shell.component';
import type { AuthUser } from '@core/auth/auth.models';

@Component({
  selector: 'app-mode-f-page',
  imports: [CommonModule, ModeFExplainerShellComponent],
  templateUrl: './mode-f-page.component.html',
  styleUrl: './mode-f-page.component.css',
})
export class ModeFPageComponent implements OnInit, OnDestroy {
  user: AuthUser | null = null;
  private subs = new Subscription();

  constructor(private readonly authSvc: AuthService) {}

  ngOnInit(): void {
    this.subs.add(
      this.authSvc.user$.subscribe(u => { this.user = u; })
    );
    this.authSvc.restoreSession();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  logout(): void {
    this.authSvc.logout();
  }
}
