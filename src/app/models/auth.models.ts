export interface AuthUser {
  id: number;
  username: string;
  displayName: string;
  createdAt: string;
}

export interface AuthResponse {
  token: string;
  user: AuthUser;
}

export interface AuthRequest {
  username: string;
  password: string;
  displayName?: string;
}

