import { CommonModule } from '@angular/common';
import { Component, inject } from '@angular/core';
import { RouterLink } from '@angular/router';
import { AuthService } from '@core/auth/auth.service';
import { PlatformTopbarComponent } from '@shared/components/platform-topbar.component';

interface PortalEntry {
  id: string;
  title: string;
  label: string;
  description: string;
  route: string;
  accent: string;
  highlights: string[];
}

@Component({
  selector: 'app-home-page',
  imports: [CommonModule, RouterLink, PlatformTopbarComponent],
  templateUrl: './home-page.component.html',
  styleUrl: './home-page.component.css',
})
export class HomePageComponent {
  private readonly authSvc = inject(AuthService);
  readonly user$ = this.authSvc.user$;

  constructor() {
    void this.authSvc.restoreSession();
  }

  readonly entries: PortalEntry[] = [
    {
      id: 'A',
      title: '模式 A',
      label: '前向传播实验室',
      description: '导入图片并编辑网络结构，观察卷积核、公式和每层激活如何共同完成一次真实前向传播。',
      route: '/mode-a',
      accent: 'blue',
      highlights: ['卷积核对比', '层公式解释', 'AI 分析'],
    },
    {
      id: 'B',
      title: '模式 B',
      label: '模型训练工作台',
      description: '选择数据集和超参数，启动后端训练任务，实时观察损失、准确率、日志与测试评估。',
      route: '/mode-b',
      accent: 'green',
      highlights: ['真实训练', '指标曲线', '模型评估'],
    },
    {
      id: 'C',
      title: '模式 C',
      label: 'CNN 解释工作台',
      description: '拆解卷积窗口、卷积核乘积、中间特征图与分类依据，理解 CNN 为什么这样判断。',
      route: '/mode-c',
      accent: 'orange',
      highlights: ['卷积过程', '特征图', '模型解释'],
    },
    {
      id: 'D',
      title: '模式 D',
      label: 'Transformer 下一词预测解释',
      description: '输入文本并观察 Top-K 概率、单层单头注意力矩阵和 QKV 教学演示，理解 Transformer 如何形成下一词预测。',
      route: '/mode-d',
      accent: 'purple',
      highlights: ['Top-K 概率', '注意力矩阵', 'QKV 演示'],
    },
    {
      id: 'E',
      title: '模式 E',
      label: '反向传播沙盒',
      description: '在沙盒中训练小型神经网络，逐阶段观察前向传播、梯度回传和参数更新。对比 SGD / Momentum / Adam 优化器效果与决策边界的变化。',
      route: '/mode-e',
      accent: 'purple',
      highlights: ['梯度回传', '优化器对比', '决策边界'],
    },
    {
      id: 'F',
      title: '模式 F',
      label: 'RNN 循环神经网络',
      description: '观察简单 RNN 处理序列数据，隐状态在时间步之间传递信息，以及 BPTT 如何穿越时间反向传播梯度。',
      route: '/mode-f',
      accent: 'cyan',
      highlights: ['时间展开', '隐状态', 'BPTT'],
    },
  ];

  readonly metrics = [
    { label: '学习模式', value: '5' },
    { label: '可视化主线', value: 'Forward / Train / CNN Explain / Transformer / Museum' },
    { label: '高级能力', value: 'Web3D + AI + WebSocket + First-person Museum' },
  ];

  logout(): void {
    this.authSvc.logout();
  }
}
