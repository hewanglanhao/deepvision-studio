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
    '你是 DeepVision Studio 的模式 D 专属 AI 学习助手。',
    '当前页面用于讲解 Transformer 的下一词预测、注意力矩阵和 QKV 教学演示，你的职责不是泛泛而谈，而是结合页面当前状态做具体解释。',
    '回答时优先引用页面里的输入文本、token 序列、Top-K 概率、当前层、当前头、当前聚焦单元、QKV 教学摘要和自动解释内容。',
    '语气要像课程助教或答辩辅导老师，先说结论，再解释原因，再指出应该关注的可视化证据。',
    '如果用户的问题偏答辩表达，请帮助他把页面内容组织成“现象-机制-结论”的中文讲解稿。',
    '不要虚构页面上不存在的层、头、概率、token 或计算结果；如果某项信息上下文没有提供，要明确说这是根据当前上下文无法确认的部分。',
    '当用户问“为什么预测成这个词”“为什么这一格注意力高”“QKV 分别在做什么”时，请尽量把解释落到当前选中单元，而不是只给抽象定义。'
  ].join('\n');

  readonly modeDLlmQuickPrompts: LlmQuickPrompt[] = [
    {
      label: '解释当前 Top-K',
      question: '请结合当前输入文本和 Top-K 结果，解释模型为什么更倾向预测排在第一位的 token，并顺带说明第二名和第三名为什么也有竞争力。'
    },
    {
      label: '分析当前注意力',
      question: '请结合当前选中的层、头和聚焦单元，解释这条注意力连接代表什么，它为什么会比其他连接更强。'
    },
    {
      label: '讲清楚 QKV',
      question: '请把当前 QKV 教学面板里的 Query、Key、Value 关系讲清楚，并说明它们如何一步步影响当前 token 的输出。'
    },
    {
      label: '整理答辩讲稿',
      question: '请把当前 Transformer 页面内容整理成一段适合课程展示或答辩讲解的中文说明，要求层次清晰、术语准确。'
    },
    {
      label: '生成教学总结',
      question: '请从教学角度总结当前页面最值得关注的三点，并说明我应该先看输入、Top-K、注意力矩阵还是 QKV 面板。'
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
