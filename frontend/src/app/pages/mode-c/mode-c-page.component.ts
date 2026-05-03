import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { Subscription } from 'rxjs';
import { ModeCExplainerShellComponent } from '../../features/mode-c-explainer/components/shell/mode-c-explainer-shell.component';
import { AuthUser } from '../../models/auth.models';
import { AuthService } from '../../services/auth.service';

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
