import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterLink } from '@angular/router';
import { Subscription } from 'rxjs';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import {
  LlmFloatingAssistantComponent,
  LlmQuickPrompt
} from '@shared/llm/llm-floating-assistant.component';
import { LlmChatContext } from '@shared/llm/llm.models';
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
  imports: [CommonModule, FormsModule, RouterLink, PlatformTopbarComponent, LlmFloatingAssistantComponent],
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

  readonly llmSystemPrompt = [
    '你是 Mode G 个性化练习页面的苏格拉底式解题教练。',
    '用户会把人工智能或深度学习选择题发给你。你的目标是引导用户自己作答，而不是替用户直接作答。',
    '严格规则：不要直接说出正确选项字母，不要说“答案是 X”，不要一次性排除到只剩唯一答案，不要输出最终答案。',
    '你应该先判断题目考查的核心概念，再用 1 到 3 个循序渐进的问题引导用户回忆定义、比较选项、发现关键条件。',
    '如果用户已经选择了某个选项，先追问他选择这个选项的理由，再提示需要核对的概念。',
    '如果用户反复要求答案，也只给思路、类比、检查清单和下一步思考问题。',
    '语气要像助教，简洁、耐心，每次回复优先推动用户说出下一步判断。'
  ].join('\n');

  readonly llmQuickPrompts: LlmQuickPrompt[] = [
    {
      label: '引导当前题',
      question: '请基于当前题目进行苏格拉底式引导，不要直接告诉我正确选项。'
    },
    {
      label: '拆知识点',
      question: '请指出当前题目考查哪些核心知识点，并用提问方式引导我回忆它们。'
    },
    {
      label: '分析选项',
      question: '请引导我逐个比较选项，但不要直接排除到唯一答案。'
    },
    {
      label: '给提示',
      question: '请只给一个小提示和一个追问，帮助我继续思考当前题。'
    }
  ];

  readonly llmContextProvider = (): LlmChatContext => this.buildLlmContext();

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

  askAiForCurrentQuestion(ai: LlmFloatingAssistantComponent): void {
    const question = this.currentQuestion;
    if (!question) return;
    ai.askPreset('我想先听一个解题引导。', true);
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

  private buildLlmContext(): LlmChatContext {
    const question = this.currentQuestion;
    const selected = this.selectedOption === null ? '尚未选择' : `${this.optionLetter(this.selectedOption)}. ${question?.options[this.selectedOption] ?? ''}`;
    const profileText = this.topicViews
      .map(topic => `${topic.label}: ${topic.score} (${topic.level})`)
      .join('；');
    const questionText = question
      ? [
          `当前题号：${question.code}`,
          `题目主题：${this.topicLabels[question.topic] || question.topic}`,
          `难度：${question.difficulty} (${this.difficultyLabel(question.difficulty)})`,
          `题干：${question.prompt}`,
          '选项：',
          ...question.options.map((option, index) => `${this.optionLetter(index)}. ${option}`),
          `推荐原因：${question.recommendationReason}`,
          `用户当前选择：${selected}`,
          `是否已经提交：${this.answerResult ? '已提交' : '未提交'}`
        ].join('\n')
      : '当前没有选中的题目。';

    return {
      text: [
        '这是 Mode G 个性化出题作题页面的当前上下文。',
        '注意：上下文不包含正确答案。请只做苏格拉底式引导，不要直接输出答案。',
        questionText,
        `出题策略：${this.recommendation?.strategyName || this.selectedMode}`,
        `策略说明：${this.recommendation?.strategyDescription || '暂无'}`,
        `用户画像：${profileText || '暂无'}`
      ].join('\n\n'),
      images: []
    };
  }

  private optionLetter(index: number): string {
    return ['A', 'B', 'C', 'D', 'E'][index] ?? String(index + 1);
  }
}
