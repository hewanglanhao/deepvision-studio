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
      label: 'Transformer 架构解剖台',
      description: '用小型显式矩阵真实计算 Q/K/V、注意力、多头拼接、残差归一化和 FFN，理解 Transformer 架构原理。',
      route: '/mode-e',
      accent: 'teal',
      highlights: ['自注意力', '多头机制', 'LayerNorm'],
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
