import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import {
  QuizAnswerResponse,
  QuizApiService,
  QuizMode,
  QuizProfileResponse,
  QuizQuestionResponse,
  QuizRecommendationResponse
} from '@shared/quiz/quiz-api.service';

interface StrategyOption {
  mode: QuizMode;
  title: string;
  short: string;
  icon: string;
}

interface TopicView {
  key: string;
  label: string;
  score: number;
  level: string;
}

@Component({
  selector: 'app-mode-g-page',
  imports: [CommonModule, FormsModule, RouterLink, PlatformTopbarComponent],
  templateUrl: './mode-g-page.component.html',
  styleUrl: './mode-g-page.component.css'
})
export class ModeGPageComponent implements OnInit, OnDestroy {
  readonly strategies: StrategyOption[] = [
    {
      mode: 'weakness',
      title: '优先补弱策略',
      short: '薄弱知识点 + 最近发展区',
      icon: '!'
    },
    {
      mode: 'spaced',
      title: '间隔复习策略',
      short: '练习时间 + 掌握变化',
      icon: '↻'
    },
    {
      mode: 'exam',
      title: '套题组卷模式',
      short: '覆盖比例 + 难度比例',
      icon: '▦'
    }
  ];

  readonly topicLabels: Record<string, string> = {
    ai_foundations: 'AI 基础',
    machine_learning: '机器学习',
    neural_networks: '神经网络',
    deep_learning_training: '深度学习训练',
    convolution_vision: '卷积与视觉',
    sequence_models: '序列模型',
    evaluation_metrics: '评估指标',
    responsible_ai: '负责任 AI'
  };

  user: AuthUser | null = null;
  profile: QuizProfileResponse | null = null;
  recommendation: QuizRecommendationResponse | null = null;
  selectedMode: QuizMode = 'weakness';
  questionLimit = 10;
  currentIndex = 0;
  selectedOption: number | null = null;
  answerResult: QuizAnswerResponse | null = null;
  loading = false;
  answering = false;
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
        void this.loadAll();
      }
      if (!user) {
        this.profile = null;
        this.recommendation = null;
      }
    }));
    void this.authSvc.restoreSession();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  get questions(): QuizQuestionResponse[] {
    return this.recommendation?.questions ?? [];
  }

  get currentQuestion(): QuizQuestionResponse | null {
    return this.questions[this.currentIndex] ?? null;
  }

  get topicViews(): TopicView[] {
    const scores = this.profile?.scores ?? {};
    return Object.keys(this.topicLabels).map(key => {
      const score = Math.round(scores[key] ?? 50);
      return {
        key,
        label: this.topicLabels[key] ?? key,
        score,
        level: this.masteryLevel(score)
      };
    });
  }

  get progressText(): string {
    return this.questions.length ? `${this.currentIndex + 1} / ${this.questions.length}` : '0 / 0';
  }

  get statusPills(): string[] {
    if (!this.profile) return [];
    return [
      `已答 ${this.profile.answeredCount}`,
      `正确率 ${(this.profile.accuracy * 100).toFixed(0)}%`
    ];
  }

  async loadAll(): Promise<void> {
    if (!this.user) return;
    this.loading = true;
    this.error = '';
    try {
      const [profile, recommendation] = await Promise.all([
        this.quizApi.profile(),
        this.quizApi.recommendations(this.selectedMode, this.questionLimit)
      ]);
      this.profile = profile;
      this.recommendation = recommendation;
      this.currentIndex = 0;
      this.selectedOption = null;
      this.answerResult = null;
    } catch (err) {
      this.error = err instanceof Error ? err.message : '加载题目失败。';
    } finally {
      this.loading = false;
    }
  }

  async selectMode(mode: QuizMode): Promise<void> {
    if (this.selectedMode === mode) return;
    this.selectedMode = mode;
    await this.loadAll();
  }

  async refreshQuestions(): Promise<void> {
    await this.loadAll();
  }

  chooseOption(index: number): void {
    if (this.answerResult) return;
    this.selectedOption = index;
  }

  async submitAnswer(): Promise<void> {
    const question = this.currentQuestion;
    if (!question || this.selectedOption === null || this.answering) return;
    this.answering = true;
    this.error = '';
    try {
      const result = await this.quizApi.answer(question.code, this.selectedOption);
      this.answerResult = result;
      this.profile = result.profile;
    } catch (err) {
      this.error = err instanceof Error ? err.message : '提交答案失败。';
    } finally {
      this.answering = false;
    }
  }

  nextQuestion(): void {
    if (this.currentIndex < this.questions.length - 1) {
      this.currentIndex += 1;
      this.selectedOption = null;
      this.answerResult = null;
    }
  }

  previousQuestion(): void {
    if (this.currentIndex > 0) {
      this.currentIndex -= 1;
      this.selectedOption = null;
      this.answerResult = null;
    }
  }

  optionClass(index: number): string {
    if (!this.answerResult) {
      return this.selectedOption === index ? 'selected' : '';
    }
    if (this.answerResult.answerIndex === index) return 'correct';
    if (this.answerResult.selectedIndex === index) return 'wrong';
    return '';
  }

  difficultyLabel(level: number): string {
    return ['入门', '基础', '进阶', '挑战'][Math.max(0, Math.min(3, level - 1))] ?? '基础';
  }

  logout(): void {
    this.authSvc.logout();
  }

  private masteryLevel(score: number): string {
    if (score >= 80) return '稳固';
    if (score >= 60) return '发展';
    if (score >= 40) return '待巩固';
    return '薄弱';
  }
}
