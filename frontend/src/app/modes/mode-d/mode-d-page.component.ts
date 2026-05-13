import { Component, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import { Subscription } from 'rxjs';
import { AuthService } from '@core/auth/auth.service';
import { ModeDExplainerShellComponent } from '@modes/mode-d/explainer/components/shell/mode-d-explainer-shell.component';
import type { AuthUser } from '@core/auth/auth.models';

@Component({
  selector: 'app-mode-d-page',
  imports: [CommonModule, ModeDExplainerShellComponent],
  templateUrl: './mode-d-page.component.html',
  styleUrl: './mode-d-page.component.css',
})
export class ModeDPageComponent implements OnInit, OnDestroy {
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
