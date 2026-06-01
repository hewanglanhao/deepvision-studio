import { Component, Input, Output, EventEmitter, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import type { AuthUser } from '@core/auth/auth.models';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import { LlmFloatingAssistantComponent, LlmQuickPrompt } from '@shared/llm/llm-floating-assistant.component';
import { TeachingSearchFabComponent } from '@shared/teaching/teaching-search-fab.component';
import { LlmChatContext } from '@shared/llm/llm.models';
import { ModeFStateService } from '../../services/mode-f-state.service';
import { ModeFOverviewComponent } from '../overview/mode-f-overview.component';
import { ModeFDetailPanelComponent } from '../detail-panel/mode-f-detail-panel.component';
import { ModeFControlPanelComponent } from '../control-panel/mode-f-control-panel.component';

const MODE_F_SYSTEM_PROMPT = `你是 DeepVision Studio 模式 F（RNN 循环神经网络教学）的 AI 助手。
当前用户正在观察一个简单 RNN 处理序列数据，学习基于时序信息的分类任务。
你的回答应围绕：RNN 的前向传播（隐状态随时间更新）、BPTT（穿越时间的反向传播）、
梯度消失/爆炸问题、RNN 如何"记忆"序列中的早期信息等主题。`;

@Component({
  selector: 'app-mode-f-explainer-shell',
  imports: [CommonModule, PlatformTopbarComponent, LlmFloatingAssistantComponent, TeachingSearchFabComponent, ModeFOverviewComponent, ModeFDetailPanelComponent, ModeFControlPanelComponent],
  templateUrl: './mode-f-explainer-shell.component.html',
  styleUrl: './mode-f-explainer-shell.component.css',
})
export class ModeFExplainerShellComponent implements OnInit {
  @Input() user: AuthUser | null = null;
  @Output() logoutRequested = new EventEmitter<void>();
  constructor(readonly state: ModeFStateService) {}
  ngOnInit(): void { this.state.loadPreset(this.state.selectedPresetId()); }
  get isLoggedIn(): boolean { return !!this.user; }
  get statusPills(): string[] { const s = this.state.status(); return [s === 'running' ? '训练中' : s === 'paused' ? '已暂停' : '就绪']; }
  requestLogout(): void { this.logoutRequested.emit(); }

  readonly systemPrompt = MODE_F_SYSTEM_PROMPT;
  readonly quickPrompts: LlmQuickPrompt[] = [
    { label: '解释 RNN', question: 'RNN 和普通 MLP 有什么本质区别？为什么 RNN 能处理序列数据？' },
    { label: 'BPTT 原理', question: '什么是 BPTT？梯度是如何在时间步之间流动的？' },
    { label: '隐状态', question: 'RNN 的隐状态向量代表什么？为什么它能"记住"之前看到的信息？' },
    { label: '诊断训练', question: '当前 RNN 训练效果如何？损失和准确率趋势说明了什么？' },
  ];

  contextProvider = (): LlmChatContext => {
    const cfg = this.state.trainingConfig();
    const meta = this.state.networkMeta();
    const step = this.state.currentStep();
    const parts = [
      `网络: ${meta?.name ?? '—'} (隐层${meta?.hiddenDim ?? 4}维)`,
      `配置: ${cfg.optimizer}, lr=${cfg.learningRate}`,
      `迭代: ${this.state.currentIteration()}`,
    ];
    if (step) parts.push(`损失: ${step.loss.toFixed(4)}, 预测: ${this.state.predictedLabel()}, 真实: ${this.state.trueLabel()}`);
    return { text: parts.join('\n'), images: [] };
  };
}
