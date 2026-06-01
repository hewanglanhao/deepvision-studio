import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { Subscription } from 'rxjs';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';
import { LlmFloatingAssistantComponent, LlmQuickPrompt } from '@shared/llm/llm-floating-assistant.component';
import { LlmChatContext } from '@shared/llm/llm.models';
import { TeachingSearchFabComponent } from '@shared/teaching/teaching-search-fab.component';
import { ModeEExplainerShellComponent } from './explainer/components/shell/mode-e-explainer-shell.component';
import { ModeEStateService } from './explainer/services/mode-e-state.service';

@Component({
  selector: 'app-mode-e-page',
  imports: [
    CommonModule,
    ModeEExplainerShellComponent,
    TeachingSearchFabComponent,
    LlmFloatingAssistantComponent
  ],
  templateUrl: './mode-e-page.component.html',
  styleUrl: './mode-e-page.component.css'
})
export class ModeEPageComponent implements OnInit, OnDestroy {
  readonly modeELlmSystemPrompt = [
    '你是 DeepVision Studio 中的反向传播学习助手。',
    '当前页面是模式 E 的反向传播沙盒模块，重点解释神经网络前向传播、损失计算、梯度回传、优化器更新和决策边界变化。',
    '回答时优先结合页面上下文中的网络结构、激活值、权重、梯度、损失曲线和决策边界，使用清晰、教学化、适合答辩讲解的中文。'
  ].join('\n');

  readonly modeELlmQuickPrompts: LlmQuickPrompt[] = [
    {
      label: '解释当前梯度',
      question: '请结合当前页面的网络结构和训练状态，解释当前梯度流动的方向及其意义。'
    },
    {
      label: '分析优化器差异',
      question: '请比较当前页面中 SGD、Momentum、Adam 三种优化器的差异，说明为什么在这个任务上某种优化器可能更好。'
    },
    {
      label: '解释激活函数',
      question: '请解释当前网络中使用的激活函数的特点，包括公式、导数和决策边界形状。'
    },
    {
      label: '答辩式总结',
      question: '请把当前反向传播页面内容整理成一段适合课程答辩讲解的说明。'
    }
  ];

  readonly modeELlmContextProvider = (): LlmChatContext => {
    const step = this.state.currentStep();
    const config = this.state.trainingConfig();
    const networkName = this.state.networkMeta()?.name ?? '当前网络';
    const datasetName = this.state.datasetMeta()?.name ?? '当前数据集';

    const lines = [
      '当前页面是 DeepVision Studio 的模式 E，用于演示反向传播与参数更新。',
      `当前网络：${networkName}。数据集：${datasetName}。`,
      `优化器：${config.optimizer}，学习率：${config.learningRate}，损失函数：${config.lossFunction}。`,
      `当前迭代：${this.state.currentIteration()}。`,
      step ? `损失值：${step.loss?.toFixed(6) ?? '—'}。预测：${this.state.predictedClassLabel()}，真实：${this.state.trueClassLabel()}。` : '',
      `训练参数总数：${this.state.totalTrainableParams()}。`,
      `当前状态：${this.state.readableStatus()}。`,
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
    private readonly state: ModeEStateService
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
