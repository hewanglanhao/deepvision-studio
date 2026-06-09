import { Injectable, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class TeachingSearchService {
  readonly active = signal(false);

  /** 显式开关术语检索模式，所有带 appTeachingTerm 的深度学习概念会同步高亮或恢复。 */
  setActive(value: boolean): void {
    this.active.set(value);
  }

  /** 由悬浮按钮触发检索模式切换，让用户随时查看界面术语背后的教学解释。 */
  toggle(): void {
    this.active.update(value => !value);
  }
}
