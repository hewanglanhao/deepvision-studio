import { CommonModule } from '@angular/common';
import { Component, OnDestroy, OnInit } from '@angular/core';
import { Subscription } from 'rxjs';
import { ModeCExplainerShellComponent } from '@modes/mode-c/explainer/components/shell/mode-c-explainer-shell.component';
import { ModeCStateService } from '@modes/mode-c/explainer/services/mode-c-state.service';
import { AuthUser } from '@core/auth/auth.models';
import { AuthService } from '@core/auth/auth.service';
import { LlmFloatingAssistantComponent, LlmQuickPrompt } from '@shared/llm/llm-floating-assistant.component';
import { LlmChatContext } from '@shared/llm/llm.models';
import { MODE_C_LLM_SYSTEM_PROMPT } from '@shared/llm/llm-prompts';
import { TeachingSearchFabComponent } from '@shared/teaching/teaching-search-fab.component';

@Component({
  selector: 'app-mode-c-page',
  imports: [CommonModule, ModeCExplainerShellComponent, LlmFloatingAssistantComponent, TeachingSearchFabComponent],
  templateUrl: './mode-c-page.component.html',
  styleUrl: './mode-c-page.component.css'
})
export class ModeCPageComponent implements OnInit, OnDestroy {
  readonly modeCLlmSystemPrompt = MODE_C_LLM_SYSTEM_PROMPT;
  readonly modeCLlmContextProvider = (): LlmChatContext => this.buildModeCLlmContext();
  readonly modeCLlmQuickPrompts: LlmQuickPrompt[] = [
    {
      label: '解释当前层',
      question: '请结合当前 Mode C 页面数据，解释当前选中层在这张样例图上为什么会产生这样的响应。'
    },
    {
      label: '分析当前通道',
      question: '请结合当前选中通道、层预览和中间过程，说明这个通道在捕捉什么特征。'
    },
    {
      label: '看分类原因',
      question: '请解释当前样例为什么会得到现在的 softmax 排名，并指出哪些层最影响最终分类。'
    },
    {
      label: '讲卷积过程',
      question: '请用教学化语言讲一遍当前卷积层里 3×3 patch、kernel、逐元素乘积、累加和 bias 各自的作用。'
    },
    {
      label: '答辩总结',
      question: '请把当前 Mode C 页面正在展示的内容整理成一段适合课程答辩讲解的说明。'
    }
  ];

  user: AuthUser | null = null;

  private readonly subs = new Subscription();

  constructor(
    private readonly authSvc: AuthService,
    private readonly state: ModeCStateService
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

  private buildModeCLlmContext(): LlmChatContext {
    const sample = this.state.currentSample();
    const prediction = this.state.currentSamplePrediction();
    const layer = this.state.selectedLayer();
    const summary = this.state.selectedLayerSummary();
    const detail = this.state.selectedLayerDetail();
    const preview = this.state.selectedLayerPreview();
    const selectedChannel = this.state.selectedChannelIndex();
    const activeChannel = detail?.channelPreviews.find(channel => channel.index === selectedChannel) ?? null;
    const topClasses = prediction?.topClasses ?? [];
    const contextLines: string[] = [];

    contextLines.push('这是 DeepVision Studio 的 Mode C，可解释卷积神经网络可视化页面。');

    if (sample) {
      contextLines.push(`当前样例：${sample.title}。`);
      contextLines.push(`样例说明：${sample.label}。`);
      contextLines.push(`样例描述：${sample.description}`);
    }

    if (prediction) {
      contextLines.push(`当前预测：${prediction.label}，置信度 ${(prediction.confidence * 100).toFixed(2)}%。`);
      if (topClasses.length) {
        contextLines.push(
          `Softmax 排名前三：${topClasses
            .slice(0, 3)
            .map((candidate, index) => `${index + 1}. ${candidate.label} ${(candidate.score * 100).toFixed(2)}%`)
            .join('；')}。`
        );
      }
    }

    if (layer) {
      contextLines.push(`当前选中层：${layer.title}（类型：${layer.type}）。`);
      contextLines.push(`输入形状：${layer.inputShapeLabel}；输出形状：${layer.outputShapeLabel}。`);
      if (layer.kernelLabel) {
        contextLines.push(`卷积或池化窗口：${layer.kernelLabel}。`);
      }
      contextLines.push(`参数量：${layer.parameterCount}。`);
      contextLines.push(`层说明：${layer.description}`);
    }

    if (summary) {
      contextLines.push(
        `当前层激活统计：最小值 ${summary.min.toFixed(4)}，最大值 ${summary.max.toFixed(4)}，均值 ${summary.mean.toFixed(4)}，正值占比 ${(summary.positiveRatio * 100).toFixed(2)}%，能量 ${summary.energy.toFixed(4)}。`
      );
    }

    if (activeChannel) {
      contextLines.push(
        `当前聚焦通道：第 ${activeChannel.index} 个通道，均值 ${activeChannel.mean.toFixed(4)}，能量 ${activeChannel.energy.toFixed(4)}。`
      );
    }

    if (detail?.convExamples?.length) {
      const example =
        detail.convExamples.find(item => item.outputChannelIndex === selectedChannel) ?? detail.convExamples[0];
      contextLines.push(
        `卷积中间过程：当前输出通道由输入通道 ${example.inputChannelIndex} 的 3×3 patch 与 kernel 做逐元素乘积并累加，再加上 bias ${example.bias.toFixed(4)}，得到输出值 ${example.outputValue.toFixed(4)}。`
      );
    }

    if (detail?.reluExamples?.length) {
      const example = detail.reluExamples.find(item => item.channelIndex === selectedChannel) ?? detail.reluExamples[0];
      contextLines.push(
        `ReLU 中间过程：当前通道激活前最小值 ${example.beforeMin.toFixed(4)}，激活后最小值 ${example.afterMin.toFixed(4)}，激活后最大值 ${example.afterMax.toFixed(4)}。`
      );
    }

    if (detail?.poolExamples?.length) {
      const example = detail.poolExamples.find(item => item.channelIndex === selectedChannel) ?? detail.poolExamples[0];
      contextLines.push(
        `池化中间过程：当前示例展示的是位置 (${example.row}, ${example.col}) 的 2×2 patch，其中最大值为 ${example.maxValue.toFixed(4)}。`
      );
    }

    const images = [
      sample ? { title: `当前样例：${sample.title}`, url: sample.assetPath } : null,
      preview ? { title: layer ? `${layer.title} 层预览` : '当前层预览', url: preview.dataUrl } : null,
      activeChannel ? { title: `当前通道 ${activeChannel.index}`, url: activeChannel.dataUrl } : null
    ].filter((item): item is { title: string; url: string } => !!item);

    return {
      text: contextLines.join('\n'),
      images
    };
  }
}
