import { Injectable } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class ApiClientService {
  readonly baseUrl: string = '';
  private readonly tokenKey = 'deepvision.auth.token';

  /** 读取本地保存的 JWT，后续接口请求会用它证明当前用户身份。 */
  get token(): string {
    return localStorage.getItem(this.tokenKey) ?? '';
  }

  /** 登录或注册成功后保存后端签发的 JWT，让 A 模式历史记录等接口能绑定到当前账号。 */
  setToken(token: string): void {
    localStorage.setItem(this.tokenKey, token);
  }

  /** 退出登录时清除 JWT，避免后续请求继续携带已失效或不属于当前用户的凭证。 */
  clearToken(): void {
    localStorage.removeItem(this.tokenKey);
  }

  /** 统一发送后端 API 请求，自动补 JSON 请求头和 Bearer token，并把错误响应转换成前端可展示的异常。 */
  async request<T>(path: string, init: RequestInit = {}): Promise<T> {
    const headers = new Headers(init.headers ?? {});
    if (!headers.has('Content-Type') && init.body && typeof init.body === 'string') {
      headers.set('Content-Type', 'application/json');
    }
    if (this.token) {
      headers.set('Authorization', `Bearer ${this.token}`);
    }

    const response = await fetch(`${this.baseUrl}${path}`, { ...init, headers });
    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const body = await response.json();
        message = body?.message ?? body?.error ?? message;
      } catch {
        message = await response.text() || message;
      }
      throw new Error(message);
    }

    if (response.status === 204) {
      return undefined as T;
    }
    return response.json() as Promise<T>;
  }
}
