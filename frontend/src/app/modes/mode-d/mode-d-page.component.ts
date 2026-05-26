import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { Subscription } from 'rxjs';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';
import { LlmFloatingAssistantComponent, LlmQuickPrompt } from '@shared/llm/llm-floating-assistant.component';
import { LlmChatContext } from '@shared/llm/llm.models';
import { TeachingSearchFabComponent } from '@shared/teaching/teaching-search-fab.component';
import { ModeDExplainerShellComponent } from './explainer/components/shell/mode-d-explainer-shell.component';
import { ModeDStateService } from './explainer/services/mode-d-state.service';

@Component({
  selector: 'app-mode-d-page',
  imports: [
    CommonModule,
    ModeDExplainerShellComponent,
    TeachingSearchFabComponent,
    LlmFloatingAssistantComponent
  ],
  templateUrl: './mode-d-page.component.html',
  styleUrl: './mode-d-page.component.css'
})
export class ModeDPageComponent implements OnInit, OnDestroy {
  readonly modeDLlmSystemPrompt = [
    '你是 DeepVision Studio 中的 Transformer 学习助手。',
    '当前页面是模式 D 的 Transformer 解释模块，重点解释输入文本、下一词 Top-K 概率、单层单头注意力矩阵，以及教学版 QKV 演示。',
    '回答时优先结合页面上下文中的输入文本、token、Top-K 排名、当前层、当前头、注意力观察和 QKV 教学说明，使用清晰、教学化、适合答辩讲解的中文。'
  ].join('\n');

  readonly modeDLlmQuickPrompts: LlmQuickPrompt[] = [
    {
      label: '解释当前 Top-K',
      question: '请结合当前页面的输入文本和 Top-K 结果，解释模型为什么更倾向预测现在排在第一位的 token。'
    },
    {
      label: '分析注意力',
      question: '请结合当前选中的层和头，解释注意力矩阵中当前最强或当前聚焦的注意力连接意味着什么。'
    },
    {
      label: '解释 QKV',
      question: '请把当前 QKV 教学面板中的 Query、Key、Value 关系讲清楚，并说明它们如何影响当前 token 的输出。'
    },
    {
      label: '答辩式总结',
      question: '请把当前 Transformer 页面内容整理成一段适合课程答辩讲解的说明。'
    }
  ];

  readonly modeDLlmContextProvider = (): LlmChatContext => {
    const example = this.state.currentExample();
    const strongest = this.state.strongestAttention();
    const focus = this.state.activeAttentionDetail();
    const qkv = this.state.qkvTeaching();
    const topK = this.state.topK();
    const block = this.state.blockOptions[this.state.selectedBlockIndex()]?.label ?? '当前层';
    const head = this.state.headOptions[this.state.selectedHeadIndex()]?.label ?? '当前头';

    const lines = [
      '当前页面是 DeepVision Studio 的模式 D，用于演示 Transformer 下一词预测与注意力可视化。',
      example ? `当前样例：${example.title}。${example.subtitle}` : '',
      `当前输入：${this.state.inputText()}`,
      `当前 token 序列：${this.state.tokens().join(' | ')}`,
      `当前视角：${block}，${head}`,
      `Top-K：${topK.slice(0, 5).map(item => `${item.rank}. ${item.token} ${(item.probability * 100).toFixed(1)}%`).join('；')}`,
      `最强注意力连接：${strongest.sourceToken} -> ${strongest.targetToken}，权重 ${(strongest.weight * 100).toFixed(1)}%。`,
      `当前聚焦单元：${focus.sourceToken} -> ${focus.targetToken}，权重 ${(focus.weight * 100).toFixed(1)}%。${focus.interpretation}`,
      `QKV 教学摘要：${qkv.summary}`,
      `自动解释：${this.state.generatedExplanation()}`
    ].filter(Boolean);

    return {
      text: lines.join('\n'),
      images: []
    };
  };

  user: AuthUser | null = null;

  private readonly subs = new Subscription();

  constructor(
    private readonly authSvc: AuthService,
    private readonly state: ModeDStateService
  ) {}

  ngOnInit(): void {
    this.subs.add(this.authSvc.user$.subscribe(user => {
      this.user = user;
    }));
    void this.authSvc.restoreSession();
  }

  ngOnDestroy(): void {
    this.subs.unsubscribe();
  }

  logout(): void {
    this.authSvc.logout();
  }
}
