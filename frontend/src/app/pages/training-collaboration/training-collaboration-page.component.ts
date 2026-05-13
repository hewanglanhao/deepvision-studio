import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ActivatedRoute, Router, RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { PlatformTopbarComponent } from '../../components/platform-topbar.component';
import { AuthUser } from '../../models/auth.models';
import { AuthService } from '../../services/auth.service';
import { CollaborationMessage, CollaborationRoomSummary, CollaborationUser, TrainingCollaborationService } from '../../services/training-collaboration.service';
import { TrainingLog, TrainingRuntimeService } from '../../services/training-runtime.service';

@Component({
  selector: 'app-training-collaboration-page',
  standalone: true,
  imports: [CommonModule, FormsModule, RouterLink, PlatformTopbarComponent],
  templateUrl: './training-collaboration-page.component.html',
  styleUrl: './training-collaboration-page.component.css'
})
export class TrainingCollaborationPageComponent implements OnInit, OnDestroy {
  authUser: AuthUser | null = null;
  roomDraft = '';
  activeRoomId = '';
  chatText = '';
  error = '';
  showMentionMenu = false;

  collaborationState: 'idle' | 'connecting' | 'connected' | 'closed' | 'error' = 'idle';
  users: CollaborationUser[] = [];
  messages: CollaborationMessage[] = [];

  trainingStatus = 'idle';
  trainingEpoch = 0;
  trainingLoss = 0;
  trainingValLoss: number | null = null;
  trainingAcc = 0;
  trainingValAcc: number | null = null;
  trainingLr = 0;
  trainingTotalEpochs = 0;
  trainingCurrentBatch = 0;
  trainingTotalBatches = 0;
  trainingElapsedSeconds = 0;
  trainingEtaSeconds = 0;
  trainingGradientNorm = 0;
  trainingLogs: TrainingLog[] = [];
  activeRooms: CollaborationRoomSummary[] = [];
  roomsLoading = false;

  readonly topbarStatusPills = ['多人协作', '实时训练观察', 'WebSocket'];
  private readonly subs = new Subscription();

  constructor(
    private route: ActivatedRoute,
    private router: Router,
    private authSvc: AuthService,
    private trainingSvc: TrainingRuntimeService,
    private collaborationSvc: TrainingCollaborationService
  ) {}

  ngOnInit(): void {
    this.subs.add(this.authSvc.user$.subscribe(user => this.authUser = user));
    this.subs.add(this.collaborationSvc.state$.subscribe(state => this.collaborationState = state));
    this.subs.add(this.collaborationSvc.users$.subscribe(users => this.users = users));
    this.subs.add(this.collaborationSvc.messages$.subscribe(messages => this.messages = messages));
    this.subs.add(this.trainingSvc.logs$.subscribe(logs => this.trainingLogs = logs));
    this.subs.add(this.trainingSvc.state$.subscribe(state => {
      this.trainingStatus = state.status;
      this.trainingEpoch = state.currentEpoch;
      this.trainingTotalEpochs = state.totalEpochs ?? 0;
      this.trainingLoss = state.latestLoss;
      this.trainingValLoss = state.latestValLoss;
      this.trainingAcc = state.latestAccuracy;
      this.trainingValAcc = state.latestValAccuracy;
      this.trainingLr = state.currentLr;
      this.trainingCurrentBatch = state.currentBatch ?? 0;
      this.trainingTotalBatches = state.totalBatches ?? 0;
      this.trainingElapsedSeconds = state.elapsedSeconds;
      this.trainingEtaSeconds = state.etaSeconds;
      this.trainingGradientNorm = state.latestGradientNorm;
    }));
    this.subs.add(this.route.queryParamMap.subscribe(params => {
      const jobId = params.get('jobId')?.trim() ?? '';
      const createRoom = params.get('create') === 'true';
      if (jobId && jobId !== this.activeRoomId) {
        this.roomDraft = jobId;
        void this.connectRoom(jobId, false, createRoom);
      }
    }));
    void this.authSvc.restoreSession();
    void this.loadRooms();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
    this.collaborationSvc.disconnect();
  }

  async connectRoom(jobId = this.roomDraft, updateUrl = true, createRoom = false): Promise<void> {
    const target = jobId.trim();
    if (!target) {
      this.error = '请输入训练房间 ID，例如 train-20260513-xxxx。';
      return;
    }
    if (!createRoom) {
      const rooms = await this.loadRooms();
      if (!rooms.some(room => room.jobId === target)) {
        this.error = '该聊天室不存在。请从 B 端当前训练新建聊天室，或从现有聊天室列表中选择。';
        return;
      }
    }
    this.error = '';
    this.activeRoomId = target;
    this.roomDraft = target;
    this.trainingSvc.observeBackendJob(target);
    this.collaborationSvc.connect(target, this.authUser?.displayName ?? this.authUser?.username ?? '', createRoom);
    if (updateUrl) {
      void this.router.navigate([], {
        relativeTo: this.route,
        queryParams: { jobId: target },
        queryParamsHandling: 'merge'
      });
    }
  }

  async loadRooms(): Promise<CollaborationRoomSummary[]> {
    this.roomsLoading = true;
    try {
      this.activeRooms = await this.collaborationSvc.listRooms();
      return this.activeRooms;
    } catch (err) {
      this.error = err instanceof Error ? err.message : '加载聊天室列表失败。';
      return [];
    } finally {
      this.roomsLoading = false;
    }
  }

  joinRoom(room: CollaborationRoomSummary): void {
    this.roomDraft = room.jobId;
    void this.connectRoom(room.jobId);
  }

  send(): void {
    const text = this.chatText.trim();
    if (!text) return;
    this.collaborationSvc.send(text);
    this.chatText = '';
    this.showMentionMenu = false;
  }

  handleChatInput(): void {
    const lastToken = this.chatText.split(/\s/).at(-1) ?? '';
    this.showMentionMenu = this.collaborationState === 'connected' && lastToken.startsWith('@') && !lastToken.includes('智能助手');
  }

  handleComposerKeydown(event: KeyboardEvent): void {
    if (event.key === 'Escape') {
      this.showMentionMenu = false;
      return;
    }
    if (event.key === 'Enter') {
      event.preventDefault();
      this.send();
    }
  }

  insertAssistantMention(): void {
    const trimmedRight = this.chatText.replace(/\s*$/, '');
    const atIndex = trimmedRight.lastIndexOf('@');
    if (atIndex >= 0 && !trimmedRight.slice(atIndex).includes(' ')) {
      this.chatText = `${trimmedRight.slice(0, atIndex)}@智能助手 `;
    } else {
      this.chatText = `${trimmedRight}${trimmedRight ? ' ' : ''}@智能助手 `;
    }
    this.showMentionMenu = false;
  }

  messageClass(message: CollaborationMessage): string {
    if (message.type === 'system') return 'message system';
    return message.username === 'robot-assistant' ? 'message assistant' : 'message';
  }

  leave(): void {
    this.collaborationSvc.disconnect();
    this.activeRoomId = '';
    this.error = '';
    void this.router.navigate([], {
      relativeTo: this.route,
      queryParams: { jobId: null },
      queryParamsHandling: 'merge'
    });
  }

  logout(): void {
    this.authSvc.logout();
  }

  percent(value: number | null | undefined): string {
    return value === null || value === undefined ? 'N/A' : `${(value * 100).toFixed(1)}%`;
  }

  get trainingProgressPercent(): number {
    if (this.trainingTotalEpochs > 0) {
      return Math.max(0, Math.min(100, (this.trainingEpoch / this.trainingTotalEpochs) * 100));
    }
    if (this.trainingTotalBatches > 0) {
      return Math.max(0, Math.min(100, (this.trainingCurrentBatch / this.trainingTotalBatches) * 100));
    }
    return 0;
  }

  formatDuration(seconds: number): string {
    const safe = Math.max(0, Math.round(seconds));
    const m = Math.floor(safe / 60);
    const s = safe % 60;
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  }

  stateText(): string {
    if (this.collaborationState === 'connected') return '聊天室已连接';
    if (this.collaborationState === 'connecting') return '聊天室连接中';
    if (this.collaborationState === 'error') return '聊天室连接异常';
    return '聊天室未连接';
  }
}
