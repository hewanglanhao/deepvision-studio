import { Injectable, computed, signal } from '@angular/core';
import { ModeCModelStatusSummary } from '../models/mode-c.types';

@Injectable({ providedIn: 'root' })
export class ModeCModelService {
  private readonly initialized = signal(false);

  readonly shellStatus = computed<ModeCModelStatusSummary>(() => {
    if (this.initialized()) {
      return {
        title: '卷积解释已就绪',
        description: '当前页面已加载为可交互的 CNN 卷积过程解释视图。',
        status: 'ready'
      };
    }

    return {
      title: '正在准备',
      description: '正在初始化 Mode C 页面。',
      status: 'in-progress'
    };
  });

  initializeNativeShell(): void {
    if (this.initialized()) return;
    this.initialized.set(true);
  }
}
