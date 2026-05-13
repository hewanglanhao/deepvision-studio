import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { RouterLink } from '@angular/router';
import type { AuthUser } from '@core/auth/auth.models';

@Component({
  selector: 'app-platform-topbar',
  imports: [CommonModule, RouterLink],
  templateUrl: './platform-topbar.component.html',
  styleUrl: './platform-topbar.component.css',
})
export class PlatformTopbarComponent {
  @Input() modeLabel = '';
  @Input() modeTitle = '';
  @Input() statusPills: string[] = [];
  @Input() user: AuthUser | null = null;
  @Output() readonly logoutRequested = new EventEmitter<void>();

  logout(): void {
    this.logoutRequested.emit();
  }
}
