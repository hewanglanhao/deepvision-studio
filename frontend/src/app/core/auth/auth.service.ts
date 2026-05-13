import { Injectable } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { AuthRequest, AuthResponse, AuthUser } from '@core/auth/auth.models';
import { ApiClientService } from '@core/api/api-client.service';

@Injectable({ providedIn: 'root' })
export class AuthService {
  private readonly userKey = 'deepvision.auth.user';
  private readonly userSubject = new BehaviorSubject<AuthUser | null>(this.readStoredUser());
  readonly user$ = this.userSubject.asObservable();

  constructor(private api: ApiClientService) {}

  get currentUser(): AuthUser | null {
    return this.userSubject.value;
  }

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

  async login(username: string, password: string): Promise<void> {
    const response = await this.sendAuth('/api/auth/login', { username, password });
    this.acceptAuth(response);
  }

  async register(username: string, password: string, displayName: string): Promise<void> {
    const response = await this.sendAuth('/api/auth/register', { username, password, displayName });
    this.acceptAuth(response);
  }

  logout(): void {
    this.api.clearToken();
    localStorage.removeItem(this.userKey);
    this.userSubject.next(null);
  }

  private async sendAuth(path: string, body: AuthRequest): Promise<AuthResponse> {
    return this.api.request<AuthResponse>(path, {
      method: 'POST',
      body: JSON.stringify(body)
    });
  }

  private acceptAuth(response: AuthResponse): void {
    this.api.setToken(response.token);
    this.storeUser(response.user);
  }

  private storeUser(user: AuthUser): void {
    localStorage.setItem(this.userKey, JSON.stringify(user));
    this.userSubject.next(user);
  }

  private readStoredUser(): AuthUser | null {
    try {
      const raw = localStorage.getItem(this.userKey);
      return raw ? JSON.parse(raw) as AuthUser : null;
    } catch {
      return null;
    }
  }
}
