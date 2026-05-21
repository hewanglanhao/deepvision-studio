import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';
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
  readonly entries: PortalEntry[] = [
    {
      id: 'A',
      title: '模式 A',
      label: '前向传播实验室',
      description: '导入图片、编辑网络结构，观察卷积核、公式和每层激活如何共同完成一次真实前向传播。',
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
      label: '反向传播沙盒',
      description: '在二维数据和 MLP 上逐步观察 loss、梯度、优化器与参数更新，理解模型如何学习。',
      route: '/mode-d',
      accent: 'purple',
      highlights: ['梯度回传', '优化器对比', '决策边界'],
    },
    {
      id: 'E',
      title: '模式 E',
      label: 'Transformer 解剖台',
      description: '用小型显式矩阵真实计算 Q/K/V、注意力、多头拼接、残差归一化和 FFN，理解 Transformer 架构原理。',
      route: '/mode-e',
      accent: 'teal',
      highlights: ['自注意力', '多头机制', 'LayerNorm'],
    },
  ];

  readonly metrics = [
    { label: '学习模式', value: '5' },
    { label: '可视化主线', value: 'Forward / Train / Explain / Backprop / Museum' },
    { label: '高级能力', value: 'Web3D + AI + WebSocket + First-person Museum' },
  ];
}
