import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { Subscription } from 'rxjs';
import { ModeCExplainerShellComponent } from '@modes/mode-c/explainer/components/shell/mode-c-explainer-shell.component';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';

@Component({
  selector: 'app-mode-c-page',
  imports: [CommonModule, ModeCExplainerShellComponent],
  templateUrl: './mode-c-page.component.html',
  styleUrl: './mode-c-page.component.css'
})
export class ModeCPageComponent implements OnInit, OnDestroy {
  user: AuthUser | null = null;

  private readonly subs = new Subscription();

  constructor(private readonly authSvc: AuthService) {}

  ngOnInit(): void {
    this.subs.add(this.authSvc.user$.subscribe(user => {
      this.user = user;
    }));
    void this.authSvc.restoreSession();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  logout(): void {
    this.authSvc.logout();
  }
}
