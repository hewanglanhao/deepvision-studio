import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthUser } from '../../models/auth.models';
import { AuthService } from '../../services/auth.service';

@Component({
  selector: 'app-mode-c-page',
  imports: [CommonModule, RouterLink],
  templateUrl: './mode-c-page.component.html',
  styleUrl: './mode-c-page.component.css'
})
export class ModeCPageComponent implements OnInit, OnDestroy {
  readonly moduleUrl: SafeResourceUrl;
  readonly assetTarget = '/modules/cnn-explainer/index.html';
  readonly sourceProjectPath = 'D:\\VS Code\\cnn-explainer';
  readonly syncCommand = '.\\scripts\\sync-cnn-explainer.ps1';
  user: AuthUser | null = null;

  private readonly subs = new Subscription();

  constructor(
    private sanitizer: DomSanitizer,
    private authSvc: AuthService
  ) {
    this.moduleUrl = this.sanitizer.bypassSecurityTrustResourceUrl(this.assetTarget);
  }

  ngOnInit(): void {
    this.subs.add(this.authSvc.user$.subscribe(user => {
      this.user = user;
    }));
    void this.authSvc.restoreSession();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  get isLoggedIn(): boolean {
    return !!this.user;
  }

  logout(): void {
    this.authSvc.logout();
  }
}
