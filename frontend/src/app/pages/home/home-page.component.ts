import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { RouterLink } from '@angular/router';

interface PortalEntry {
  title: string;
  label: string;
  description: string;
  route: string;
  status: string;
}

@Component({
  selector: 'app-home-page',
  imports: [CommonModule, RouterLink],
  templateUrl: './home-page.component.html',
  styleUrl: './home-page.component.css'
})
export class HomePageComponent {
  readonly entries: PortalEntry[] = [
    {
      title: '模式 A',
      label: '前向传播可视化',
      description: '导入图片、编辑网络层、查看张量尺寸和每层输出。',
      route: '/mode-a',
      status: '已接入'
    },
    {
      title: '模式 B',
      label: '训练工作台',
      description: '选择数据集、配置训练参数、观察指标曲线和训练日志。',
      route: '/mode-b',
      status: '已接入'
    },
    {
      title: '模式 C',
      label: '预留业务端',
      description: '后续可独立增加路由、页面组件和专属服务。',
      route: '/mode-c',
      status: '已占位'
    },
    {
      title: '模式 D',
      label: '反向传播可视化',
      description: '观察前向传播、损失计算、反向传播和参数更新的完整过程。',
      route: '/mode-d',
      status: '已接入'
    }
  ];
}
