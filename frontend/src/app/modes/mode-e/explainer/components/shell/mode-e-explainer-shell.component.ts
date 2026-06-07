import { Component, Input, Output, EventEmitter, OnInit, OnDestroy } from '@angular/core';
import { CommonModule } from '@angular/common';
import type { AuthUser } from '@core/auth/auth.models';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';
import { LlmFloatingAssistantComponent, LlmQuickPrompt } from '@shared/llm/llm-floating-assistant.component';
import { TeachingSearchFabComponent } from '@shared/teaching/teaching-search-fab.component';
import { LlmChatContext } from '@shared/llm/llm.models';
import { ModeEOverviewComponent } from '../overview/mode-e-overview.component';
import { ModeEDetailPanelComponent } from '../detail-panel/mode-e-detail-panel.component';
import { ModeEControlPanelComponent } from '../control-panel/mode-e-control-panel.component';
import { ModeEFloatingChartsComponent } from '../floating-charts/mode-e-floating-charts.component';
import { ModeEStateService } from '../../services/mode-e-state.service';

const MODE_D_LLM_SYSTEM_PROMPT = `你是 DeepVision Studio 模式 E 的 AI 教学助手。当前用户在观察一个 MLP 在二维数据上进行反向传播训练的完整过程。

你可以看到的数据包括：网络结构（层数、神经元数）、训练配置（优化器类型、学习率）、各层梯度范数、权重变化量、损失值、决策边界变化等。所有计算在浏览器内用纯 TypeScript 完成，单样本随机训练，每 25 步用全量样本评估一次平均损失和准确率。

你的回答应围绕以下内容展开，风格专业但有教学性：

1. **优化器行为分析**：
   - 为何 SGD 在 XOR 上表现好却在高斯团/同心圆上差？—— XOR 损失曲面简单光滑，SGD 的随机性有助于快速收敛；高斯团/同心圆曲面复杂有平坦区，SGD 噪声大易卡住，Adam 的自适应学习率和动量更适合穿过这些区域
   - 不同优化器的收敛速度对比：SGD 慢但稳，Momentum 平滑加速，Adam 快但可能过度调节
   - 如何从梯度范数判断优化器健康状态（消失<0.01，稳定0.01-5，爆炸>5）

2. **数据集特性分析**：
   - XOR 需要至少一个隐藏层的非线性网络才能分开四团数据
   - 同心圆需要环形决策边界，比 XOR 更难，Tanh 激活有助于学习对称模式
   - 螺旋线是经典非线性测试，深层网络才能拟合
   - 高斯团线性可分度较高，Sigmoid 足够处理

3. **训练诊断**：
   - 损失不下降可能原因：学习率过高/过低、网络容量不足、数据本身有噪声重叠
   - 准确率提升但损失不降说明模型在改进分类但置信度还不高
   - 梯度消失/爆炸的征兆和应对方法
   - 决策边界不平滑说明训练不充分或正则化缺失

4. **参数变化解读**：
   - 权重变化量大的层在积极学习，变化趋零说明梯度已接近最优
   - 偏置变化反映神经元激活阈值的调整

用中文回答，适当使用类比帮助理解。对于用户看到的反常现象（如某优化器在某数据集上表现意外好或差），结合数据特性和优化器原理给予解释。`;

@Component({
  selector: 'app-mode-e-explainer-shell',
  imports: [
    CommonModule,
    PlatformTopbarComponent,
    LlmFloatingAssistantComponent,
    TeachingSearchFabComponent,
    ModeEOverviewComponent,
    ModeEDetailPanelComponent,
    ModeEControlPanelComponent,
    ModeEFloatingChartsComponent,
  ],
  templateUrl: './mode-e-explainer-shell.component.html',
  styleUrl: './mode-e-explainer-shell.component.css',
})
export class ModeEExplainerShellComponent implements OnInit, OnDestroy {
  @Input() user: AuthUser | null = null;
  @Output() readonly logoutRequested = new EventEmitter<void>();

  constructor(readonly state: ModeEStateService) {}

  ngOnInit(): void {
    this.state.loadPreset(this.state.selectedPresetId());
  }

  ngOnDestroy(): void {
    this.state.pause();
  }

  get isLoggedIn(): boolean {
    return !!this.user;
  }

  get statusPills(): string[] {
    return [this.state.readableStatus()];
  }

  requestLogout(): void {
    this.logoutRequested.emit();
  }

  // ---- LLM Assistant config ------------------------------------------------

  readonly modeDLlmSystemPrompt = MODE_D_LLM_SYSTEM_PROMPT;

  readonly modeDLlmQuickPrompts: LlmQuickPrompt[] = [
    { label: '分析当前训练', question: '请根据当前的网络结构、数据集和优化器配置，分析训练表现。损失和准确率趋势如何？梯度是否健康？有什么改进建议？' },
    { label: '对比优化器', question: '当前使用了哪种优化器？如果换成另外两种（SGD/Momentum/Adam），在当前数据集上预计会有什么不同？为什么有的优化器在简单数据集上更好、在复杂数据集上反而差？' },
    { label: '诊断问题', question: '如果当前损失不下降或准确率低，可能的原因是什么？请从学习率、网络容量、数据集难度、梯度健康状态等角度综合分析。' },
    { label: '解读梯度', question: '当前各层的梯度范数是多少？这个值正常吗？为什么深层网络的梯度容易消失或爆炸？ReLU/Tanh/Sigmoid 激活函数对梯度流动有什么影响？' },
    { label: '损失vs准确率', question: '为什么有时损失不降但准确率上升？或者反之？这两个指标各反映了模型的什么特性？在分类任务中应该更关注哪个？' },
    { label: '反向传播原理', question: '请通俗解释反向传播的完整流程：前向传播算出预测值后，如何通过链式法则把输出层的误差一层层回传，最终更新所有层的权重？' },
  ];

  modeDLlmContextProvider = (): LlmChatContext => {
    return this.buildModeELlmContext();
  };

  private buildModeELlmContext(): LlmChatContext {
    const preset = this.state.networkMeta();
    const config = this.state.trainingConfig();
    const step = this.state.currentStep();
    const dataset = this.state.datasetMeta();
    const layers = this.state.networkLayers();
    const counts = this.state.neuronCounts();
    const avgLossHistory = this.state.avgLossHistory();
    const savedCurves = this.state.savedCurves();
    const accuracy = this.state.latestAccuracy();

    const parts: string[] = [];

    parts.push(`=== 当前训练状态 ===`);
    parts.push(`网络: ${preset?.name ?? '自定义'} — ${preset?.description ?? ''}`);
    parts.push(`结构: ${counts.join(' → ')} (${layers.length}层, ${this.state.totalTrainableParams()}个可训练参数)`);
    parts.push(`配置: 优化器=${config.optimizer}, 学习率=${config.learningRate}, 损失函数=${config.lossFunction}`);
    parts.push(`数据集: ${dataset?.name ?? '未知'} — ${dataset?.description ?? ''} (${this.state.currentDataset().length}个样本)`);
    parts.push(`训练进度: 当前第${this.state.currentIteration()}步`);

    if (accuracy > 0) {
      parts.push(`全量评估: 平均损失=${avgLossHistory[avgLossHistory.length-1]?.loss.toFixed(4) ?? '—'}, 整体准确率=${(accuracy*100).toFixed(1)}%`);
      if (avgLossHistory.length >= 2) {
        const first = avgLossHistory[0].loss;
        const last = avgLossHistory[avgLossHistory.length - 1].loss;
        const trend = last < first ? '下降' : '上升';
        parts.push(`损失趋势: ${first.toFixed(4)} → ${last.toFixed(4)} (${trend} ${Math.abs((1-last/first)*100).toFixed(0)}%)`);
      }
    }

    if (step) {
      parts.push(`\n=== 当前步骤详情 ===`);
      parts.push(`预测: ${this.state.predictedClassLabel()}, 真实: ${this.state.trueClassLabel()}, 单步损失: ${step.loss?.toFixed(6) ?? '—'}`);
      if (step.predictions) {
        parts.push(`输出概率: ${step.predictions.map((p, i) => `${dataset?.classLabels[i] ?? '类'+i}=${(p*100).toFixed(1)}%`).join(', ')}`);
      }
      const gradNorms = step.layerGradients.map((g, i) => {
        const status = g.gradientNorm < 0.01 ? '消失' : g.gradientNorm > 5 ? '爆炸' : '正常';
        return `${layers[i]?.name ?? 'L'+i}: ${g.gradientNorm.toFixed(4)} (${status})`;
      }).join('; ');
      parts.push(`各层梯度范数: ${gradNorms}`);

      for (const snap of step.parameterSnapshots) {
        if (snap.weightChange && snap.weightChange.length > 0 && snap.weightChange[0].length > 0) {
          const layer = layers.find(l => l.id === snap.layerId);
          const flat = snap.weightChange.flat();
          const maxChange = Math.max(...flat);
          const avgChange = flat.reduce((a, b) => a + b, 0) / flat.length;
          parts.push(`${layer?.name ?? '层'}: 最大Δw=${maxChange.toFixed(4)}, 平均Δw=${avgChange.toFixed(5)}`);
        }
      }
    }

    // Optimizer comparison data
    if (savedCurves.length > 0) {
      parts.push(`\n=== 优化器对比 ===`);
      for (const c of savedCurves) {
        const lastPt = c.points[c.points.length - 1];
        parts.push(`${c.label}: 最终损失=${lastPt.loss.toFixed(4)}, 准确率=${((lastPt.accuracy ?? 0)*100).toFixed(0)}%`);
      }
    }

    return { text: parts.join('\n'), images: [] };
  }
}
