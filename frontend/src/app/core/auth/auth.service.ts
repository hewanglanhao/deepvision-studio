import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { AuthRequest, AuthResponse, AuthUser } from '@core/auth/auth.models';
import { ApiClientService } from '@core/api/api-client.service';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly userKey = 'deepvision.auth.user';
  private readonly userSubject = new BehaviorSubject<AuthUser | null>(this.readStoredUser());
  readonly user$ = this.userSubject.asObservable();

  /** 注入 API 客户端，认证服务通过它发送登录请求并把 JWT 写入所有后续请求。 */
  constructor(private api: ApiClientService) {}

  /** 返回当前登录用户，页面据此判断能否保存或读取个人的 forward 推理记录。 */
  get currentUser(): AuthUser | null {
    return this.userSubject.value;
  }

  /** 页面刷新后用本地 JWT 向后端校验会话，JWT 失效时主动清空前端登录状态。 */
  async restoreSession(): Promise<void> {
    if (!this.api.token) return;
    try {
      const user = await this.api.request<AuthUser | null>('/api/auth/me');
      if (!user) {
        this.logout();
        return;
      }
      this.storeUser(user);
    } catch {
      this.logout();
    }
  }

  /** 校验用户名密码并签发 JWT 登录凭证。 */
  async login(username: string, password: string): Promise<void> {
    const response = await this.sendAuth('/api/auth/login', { username, password });
    this.acceptAuth(response);
  }

  /** 创建新用户、加密密码并签发登录凭证。 */
  async register(username: string, password: string, displayName: string): Promise<void> {
    const response = await this.sendAuth('/api/auth/register', { username, password, displayName });
    this.acceptAuth(response);
  }

  /** 清除本地 token 和用户缓存，退出后 A 模式不再访问该用户的私有记录。 */
  logout(): void {
    this.api.clearToken();
    localStorage.removeItem(this.userKey);
    this.userSubject.next(null);
  }

  /** 把登录和注册都收敛成同一种认证请求，统一接收后端返回的 JWT 与用户摘要。 */
  private async sendAuth(path: string, body: AuthRequest): Promise<AuthResponse> {
    return this.api.request<AuthResponse>(path, {
      method: 'POST',
      body: JSON.stringify(body)
    });
  }

  /** 接收认证成功结果：先保存 JWT，再发布用户信息让导航栏和 A 模式同步更新。 */
  private acceptAuth(response: AuthResponse): void {
    this.api.setToken(response.token);
    this.storeUser(response.user);
  }

  /** 缓存用户摘要并推送给订阅者，避免每个组件都重复请求 /api/auth/me。 */
  private storeUser(user: AuthUser): void {
    localStorage.setItem(this.userKey, JSON.stringify(user));
    this.userSubject.next(user);
  }

  /** 从 localStorage 恢复用户摘要，用作应用启动时的初始登录态。 */
  private readStoredUser(): AuthUser | null {
    try {
      const raw = localStorage.getItem(this.userKey);
      return raw ? JSON.parse(raw) as AuthUser : null;
    } catch {
      return null;
    }
  }
}
