import { Injectable, OnDestroy } from '@angular/core';
import { BehaviorSubject } from 'rxjs';
import { ApiClientService } from './api-client.service';

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
  | CollaborationMessage;

@Injectable({ providedIn: 'root' })
export class TrainingCollaborationService implements OnDestroy {
  private socket: WebSocket | null = null;
  private roomJobId = '';

  readonly messages$ = new BehaviorSubject<CollaborationMessage[]>([]);
  readonly users$ = new BehaviorSubject<CollaborationUser[]>([]);
  readonly state$ = new BehaviorSubject<'idle' | 'connecting' | 'connected' | 'closed' | 'error'>('idle');

  constructor(private api: ApiClientService) {}

  get currentRoomJobId(): string {
    return this.roomJobId;
  }

  connect(jobId: string, displayName = '', createRoom = false): void {
    const room = jobId.trim();
    if (!room) return;
    if (this.socket && this.roomJobId === room && this.state$.value === 'connected') {
      return;
    }
    this.disconnect();
    this.roomJobId = room;
    this.messages$.next([]);
    this.users$.next([]);
    this.state$.next('connecting');

    const params = new URLSearchParams({ jobId: room });
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

  async listRooms(): Promise<CollaborationRoomSummary[]> {
    return this.api.request<CollaborationRoomSummary[]>('/api/training/collaboration/rooms');
  }

  send(text: string): void {
    const content = text.trim();
    if (!content || !this.socket || this.socket.readyState !== WebSocket.OPEN) {
      return;
    }
    this.socket.send(JSON.stringify({ type: 'chat', text: content }));
  }

  disconnect(): void {
    if (this.socket) {
      const socket = this.socket;
      this.socket = null;
      socket.onclose = null;
      socket.close();
    }
    this.roomJobId = '';
    this.users$.next([]);
    if (this.state$.value !== 'idle') {
      this.state$.next('closed');
    }
  }

  ngOnDestroy(): void {
    this.disconnect();
  }

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
    }
  }

  private wsBaseUrl(): string {
    return this.api.baseUrl.replace(/^http/i, 'ws');
  }
}
