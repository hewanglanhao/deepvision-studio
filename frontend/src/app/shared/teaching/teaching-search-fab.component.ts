import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { TeachingSearchService } from './teaching-search.service';

@Component({
  selector: 'app-teaching-search-fab',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="teaching-fab-dock">
      @if (teachingSearch.active()) {
        <div class="teaching-hint">
          悬停高亮术语，点击进入教学文档
          <button type="button" (click)="openTeachingDoc()">打开文档</button>
        </div>
      }
      <button
        type="button"
        class="teaching-fab"
        [class.active]="teachingSearch.active()"
        [attr.aria-pressed]="teachingSearch.active()"
        aria-label="切换术语检索模式"
        title="术语检索"
        (click)="teachingSearch.toggle()"
      >
        ?
      </button>
    </div>
  `,
  styles: [`
    :host {
      position: fixed;
      right: 20px;
      bottom: 76px;
      z-index: 1780;
      font-family: 'Inter', 'Segoe UI', 'Noto Sans SC', sans-serif;
    }

    .teaching-fab-dock {
      position: relative;
      display: grid;
      justify-items: end;
      gap: 8px;
    }

    .teaching-fab {
      width: 46px;
      height: 46px;
      border: 1px solid #0f766e;
      border-radius: 50%;
      background: #ffffff;
      color: #0f766e;
      font-size: 21px;
      font-weight: 900;
      line-height: 1;
      box-shadow: 0 10px 24px rgba(15, 118, 110, .18);
      cursor: pointer;
      transition: transform .16s ease, background .16s ease, color .16s ease, box-shadow .16s ease;
    }

    .teaching-fab:hover,
    .teaching-fab.active {
      transform: translateY(-1px);
      background: #0f766e;
      color: #ffffff;
      box-shadow: 0 14px 30px rgba(15, 118, 110, .28);
    }

    .teaching-hint {
      width: 228px;
      padding: 8px 10px;
      border: 1px solid #99f6e4;
      border-radius: 8px;
      background: #f0fdfa;
      color: #115e59;
      font-size: 12px;
      line-height: 1.45;
      box-shadow: 0 10px 26px rgba(15, 23, 42, .12);
    }

    .teaching-hint button {
      display: inline-block;
      margin-left: 6px;
      border: 0;
      background: transparent;
      color: #0f766e;
      font: inherit;
      font-weight: 800;
      text-decoration: none;
      cursor: pointer;
      padding: 0;
    }

    @media (max-width: 430px) {
      :host {
        right: 12px;
        bottom: 68px;
      }
    }
  `]
})
export class TeachingSearchFabComponent {
  /** 注入术语检索状态，悬浮按钮用它控制整页深度学习术语是否高亮。 */
  constructor(public readonly teachingSearch: TeachingSearchService) {}

  /** 直接打开完整教学文档，适合用户不从某个具体术语进入时查阅背景知识。 */
  openTeachingDoc(): void {
    const target = window.open('/teaching', 'deepvision-teaching-docs');
    target?.focus();
  }
}
