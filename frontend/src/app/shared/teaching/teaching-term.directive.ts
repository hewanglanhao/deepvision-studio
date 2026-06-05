import { Directive, HostBinding, HostListener, Input } from '@angular/core';
import { findTeachingTerm } from './teaching-glossary';
import { TeachingSearchService } from './teaching-search.service';

const TEACHING_WINDOW_NAME = 'deepvision-teaching-docs';

@Directive({
  selector: '[appTeachingTerm]',
  standalone: true
})
export class TeachingTermDirective {
  @Input('appTeachingTerm') termId = '';

  /** 注入术语检索状态，让页面上的深度学习概念能在检索模式下变成可点击入口。 */
  constructor(
    public readonly teachingSearch: TeachingSearchService
  ) {}

  @HostBinding('class.teaching-term')
  readonly isTeachingTerm = true;

  @HostBinding('class.teaching-term-active')
  /** 根据全局检索开关决定当前术语是否高亮，帮助用户在 A 模式界面中快速定位概念。 */
  get isActive(): boolean {
    return this.teachingSearch.active();
  }

  @HostBinding('attr.tabindex')
  /** 检索模式开启时把术语加入键盘焦点序列，保证不用鼠标也能打开教学文档。 */
  get tabindex(): string | null {
    return this.isActive ? '0' : null;
  }

  @HostBinding('attr.title')
  /** 给高亮术语提供浏览器提示，说明点击后会跳到对应的教学条目。 */
  get title(): string | null {
    const term = findTeachingTerm(this.termId);
    return this.isActive && term ? `查看教学文档：${term.title}` : null;
  }

  @HostListener('click', ['$event'])
  /** 用户点击术语时打开对应教学文档，例如卷积、池化、激活函数等背景说明。 */
  onClick(event: MouseEvent): void {
    if (!this.isActive || !this.termId) return;
    event.preventDefault();
    event.stopPropagation();
    this.teachingSearch.setActive(false);
    this.openTeachingDoc();
  }

  @HostListener('keydown.enter', ['$event'])
  /** 支持回车键打开术语文档，保持鼠标点击和键盘访问行为一致。 */
  onEnter(event: Event): void {
    this.openFromKeyboard(event);
  }

  @HostListener('keydown.space', ['$event'])
  /** 支持空格键打开术语文档，符合可访问组件的常见键盘交互习惯。 */
  onSpace(event: Event): void {
    this.openFromKeyboard(event);
  }

  /** 统一处理键盘触发逻辑：阻止页面滚动或表单默认行为后跳转到术语说明。 */
  private openFromKeyboard(event: Event): void {
    if (!this.isActive || !this.termId) return;
    event.preventDefault();
    this.teachingSearch.setActive(false);
    this.openTeachingDoc();
  }

  /** 在固定窗口中打开教学页并定位到术语锚点，避免用户离开当前实验画布。 */
  private openTeachingDoc(): void {
    const url = `/teaching#${encodeURIComponent(this.termId)}`;
    const target = window.open(url, TEACHING_WINDOW_NAME);
    target?.focus();
  }
}
