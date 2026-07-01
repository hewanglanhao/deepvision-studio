import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { ApiClientService } from '@core/api/api-client.service';

export interface CollaborationUser {
  username: string;
  displayName: string;
}

export interface CollaborationMessage {
  type: 'chat' | 'system';
  id?: string;
  jobId: string;
  username?: string;
  displayName?: string;
  text: string;
  createdAt: string;
  streaming?: boolean;
}

export interface CollaborationRoomSummary {
  jobId: string;
  onlineCount: number;
  createdAt: string;
  users: string[];
}

type CollaborationInbound =
  | { type: 'history'; jobId: string; messages: CollaborationMessage[] }
  | { type: 'presence'; jobId: string; users: CollaborationUser[] }
  | { type: 'chat_update'; id: string; jobId: string; text: string; streaming?: boolean; createdAt?: string }
  | CollaborationMessage;

@Injectable({ providedIn: 'root' })
export class TrainingCollaborationService implements OnDestroy {
  private socket: WebSocket | null = null;
  private roomJobId = '';
  private clientId = '';

  readonly messages$ = new BehaviorSubject<CollaborationMessage[]>([]);
  readonly users$ = new BehaviorSubject<CollaborationUser[]>([]);
  readonly state$ = new BehaviorSubject<'idle' | 'connecting' | 'connected' | 'closed' | 'error'>('idle');

  constructor(private api: ApiClientService) {}

  get currentRoomJobId(): string {
    return this.roomJobId;
  }

  get currentClientId(): string {
    return this.clientId;
  }

  // 建立训练协作聊天室连接；createRoom 为 true 时允许后端创建新房间。
  connect(jobId: string, displayName = '', createRoom = false): void {
    const room = jobId.trim();
    if (!room) return;
    if (this.socket && this.roomJobId === room && this.state$.value === 'connected') {
      return;
    }
    this.disconnect();
    this.roomJobId = room;
    this.clientId = this.createClientId();
    this.messages$.next([]);
    this.users$.next([]);
    this.state$.next('connecting');

    const params = new URLSearchParams({ jobId: room, clientId: this.clientId });
    if (this.api.token) params.set('token', this.api.token);
    if (displayName) params.set('name', displayName);
    if (createRoom) params.set('create', 'true');
    const socket = new WebSocket(`${this.wsBaseUrl()}/api/training/collaboration?${params.toString()}`);
    this.socket = socket;
    socket.onopen = () => this.state$.next('connected');
    socket.onmessage = event => this.handleMessage(event.data);
    socket.onerror = () => this.state$.next('error');
    socket.onclose = () => {
      if (this.socket === socket) {
        this.socket = null;
      }
      if (this.state$.value !== 'error') {
        this.state$.next('closed');
      }
    };
  }

  // 查询后端当前仍然活跃的训练协作房间列表。
  async listRooms(): Promise<CollaborationRoomSummary[]> {
    return this.api.request<CollaborationRoomSummary[]>('/api/training/collaboration/rooms');
  }

  // 向当前聊天室发送文本消息，空内容或未连接时直接忽略。
  send(text: string): void {
    const content = text.trim();
    if (!content || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    this.socket.send(JSON.stringify({ type: 'chat', text: content }));
  }

  // 主动断开聊天室 WebSocket，并清空当前房间标识。
  disconnect(): void {
    if (this.socket) {
      const socket = this.socket;
      this.socket = null;
      socket.onclose = null;
      socket.close();
    }
    this.roomJobId = '';
    this.clientId = '';
    this.users$.next([]);
    if (this.state$.value !== 'idle') {
      this.state$.next('closed');
    }
  }

  // 服务销毁时关闭聊天室连接，避免页面离开后仍保留在线成员。
  ngOnDestroy(): void {
    this.disconnect();
  }

  // 解析聊天室后端消息，并按 history/presence/chat/assistant_update 更新状态流。
  private handleMessage(raw: string): void {
    let payload: CollaborationInbound;
    try {
      payload = JSON.parse(raw) as CollaborationInbound;
    } catch {
      return;
    }
    if (payload.type === 'history') {
      this.messages$.next((payload.messages ?? []).slice(-100));
      return;
    }
    if (payload.type === 'presence') {
      this.users$.next(payload.users ?? []);
      return;
    }
    if (payload.type === 'chat' || payload.type === 'system') {
      this.messages$.next([...this.messages$.value, payload].slice(-100));
      return;
    }
    if (payload.type === 'chat_update') {
      const next = this.messages$.value.map(message => {
        if (message.id !== payload.id) return message;
        return {
          ...message,
          text: payload.text,
          streaming: payload.streaming ?? false,
          createdAt: payload.createdAt ?? message.createdAt
        };
      });
      this.messages$.next(next);
    }
  }

  // 根据 API baseUrl 和当前页面协议推导 WebSocket 基础地址。
  private wsBaseUrl(): string {
    if (this.api.baseUrl) {
      return this.api.baseUrl.replace(/^http/i, 'ws');
    }
    return `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`;
  }

  // 生成浏览器端临时 clientId，供协作旁观流校验房间成员身份。
  private createClientId(): string {
    return globalThis.crypto?.randomUUID?.()
      ?? `collab-${Date.now()}-${Math.random().toString(16).slice(2)}`;
  }
}
