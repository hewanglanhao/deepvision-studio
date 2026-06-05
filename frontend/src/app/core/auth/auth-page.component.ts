import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';

type AuthPageMode = 'login' | 'register';

@Component({
  selector: 'app-auth-page',
  imports: [CommonModule, FormsModule, RouterLink, PlatformTopbarComponent],
  templateUrl: './auth-page.component.html',
  styleUrl: './auth-page.component.css',
})
export class AuthPageComponent implements OnInit, OnDestroy {
  mode: AuthPageMode = 'login';
  draft = { username: '', password: '', displayName: '' };
  busy = false;
  error = '';
  user: AuthUser | null = null;

  private readonly subs = new Subscription();

  /** 注入路由和认证服务，页面根据路由切换登录/注册模式，并在成功后回到 A 模式。 */
  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private authSvc: AuthService,
  ) {}

  /** 初始化页面状态、订阅数据源并触发首次数据加载。 */
  ngOnInit(): void {
    this.subs.add(this.route.data.subscribe(data => {
      this.mode = (data['mode'] as AuthPageMode | undefined) ?? 'login';
      this.error = '';
    }));
    this.subs.add(this.authSvc.user$.subscribe(user => { this.user = user; }));
  }

  /** 释放组件订阅、定时器和渲染资源，避免页面离开后继续占用内存。 */
  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  /** 根据当前路由模式给认证表单显示“登录”或“注册”的页面标题。 */
  get title(): string {
    return this.mode === 'login' ? '登录账号' : '注册账号';
  }

  /** 根据当前模式决定提交按钮文案，避免登录和注册共用表单时误导用户。 */
  get submitText(): string {
    return this.mode === 'login' ? '登录' : '注册';
  }

  /** 提交登录或注册表单，认证成功后进入 A 模式继续进行神经网络搭建和推理记录保存。 */
  async submit(): Promise<void> {
    if (this.busy) {
      return;
    }
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

  /** 从认证页也允许退出当前账号，便于切换到其他用户重新保存实验记录。 */
  logout(): void {
    this.authSvc.logout();
  }
}
