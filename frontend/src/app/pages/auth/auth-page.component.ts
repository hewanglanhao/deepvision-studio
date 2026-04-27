import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthUser } from '../../models/auth.models';
import { AuthService } from '../../services/auth.service';

type AuthPageMode = 'login' | 'register';

@Component({
  selector: 'app-auth-page',
  imports: [CommonModule, FormsModule, RouterLink],
  templateUrl: './auth-page.component.html',
  styleUrl: './auth-page.component.css'
})
export class AuthPageComponent implements OnInit, OnDestroy {
  mode: AuthPageMode = 'login';
  draft = { username: '', password: '', displayName: '' };
  busy = false;
  error = '';
  user: AuthUser | null = null;

  private readonly subs = new Subscription();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private authSvc: AuthService
  ) {}

  ngOnInit(): void {
    this.subs.add(this.route.data.subscribe(data => {
      this.mode = (data['mode'] as AuthPageMode | undefined) ?? 'login';
      this.error = '';
    }));
    this.subs.add(this.authSvc.user$.subscribe(user => this.user = user));
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  get title(): string {
    return this.mode === 'login' ? '登录账号' : '注册账号';
  }

  get submitText(): string {
    return this.mode === 'login' ? '登录' : '注册';
  }

  async submit(): Promise<void> {
    if (this.busy) return;
    this.busy = true;
    this.error = '';

    try {
      if (this.mode === 'login') {
        await this.authSvc.login(this.draft.username, this.draft.password);
      } else {
        await this.authSvc.register(this.draft.username, this.draft.password, this.draft.displayName);
      }
      await this.router.navigateByUrl('/mode-a');
    } catch (err) {
      this.error = err instanceof Error ? err.message : '认证请求失败，请检查后端服务。';
    } finally {
      this.busy = false;
    }
  }

  logout(): void {
    this.authSvc.logout();
  }
}
