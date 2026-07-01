import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import {
  QuizApiService,
  QuizDashboardResponse,
  QuizReviewStatus,
  QuizWeakTopic,
  QuizWrongQuestion
} from '@shared/quiz/quiz-api.service';

@Component({
  selector: 'app-mode-h-page',
  imports: [CommonModule, RouterLink, PlatformTopbarComponent],
  templateUrl: './mode-h-page.component.html',
  styleUrl: './mode-h-page.component.css'
})
export class ModeHPageComponent implements OnInit, OnDestroy {
  user: AuthUser | null = null;
  dashboard: QuizDashboardResponse | null = null;
  loading = false;
  error = '';

  private readonly subs = new Subscription();

  constructor(
    private readonly authSvc: AuthService,
    private readonly quizApi: QuizApiService
  ) {}

  ngOnInit(): void {
    this.subs.add(this.authSvc.user$.subscribe(user => {
      const changed = user?.username !== this.user?.username;
      this.user = user;
      if (user && changed) {
        void this.loadDashboard();
      }
      if (!user) {
        this.dashboard = null;
      }
    }));
    void this.authSvc.restoreSession();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  get statusPills(): string[] {
    const profile = this.dashboard?.profile;
    if (!profile) return [];
    return [
      `已答 ${profile.answeredCount}`,
      `正确率 ${(profile.accuracy * 100).toFixed(0)}%`,
      `连续正确 ${profile.currentStreak}`
    ];
  }

  get weakTopics(): QuizWeakTopic[] {
    return this.dashboard?.weakTopics ?? [];
  }

  get reviewRows(): QuizReviewStatus[] {
    return this.dashboard?.reviewStatus ?? [];
  }

  get wrongQuestions(): QuizWrongQuestion[] {
    return this.dashboard?.wrongQuestions ?? [];
  }

  async loadDashboard(): Promise<void> {
    if (!this.user) return;
    this.loading = true;
    this.error = '';
    try {
      this.dashboard = await this.quizApi.dashboard();
    } catch (err) {
      this.error = err instanceof Error ? err.message : '加载学习看板失败。';
    } finally {
      this.loading = false;
    }
  }

  scoreLevel(score: number): string {
    if (score >= 80) return '稳固';
    if (score >= 60) return '发展';
    if (score >= 40) return '待巩固';
    return '薄弱';
  }

  reviewClass(status: string): string {
    if (status === '需要复习') return 'danger';
    if (status === '到期复习') return 'warning';
    if (status === '建议巩固') return 'notice';
    return 'ok';
  }

  optionLabel(index: number): string {
    return ['A', 'B', 'C', 'D', 'E'][index] ?? String(index + 1);
  }

  logout(): void {
    this.authSvc.logout();
  }
}
