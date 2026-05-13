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

  constructor(
    public readonly teachingSearch: TeachingSearchService
  ) {}

  @HostBinding('class.teaching-term')
  readonly isTeachingTerm = true;

  @HostBinding('class.teaching-term-active')
  get isActive(): boolean {
    return this.teachingSearch.active();
  }

  @HostBinding('attr.tabindex')
  get tabindex(): string | null {
    return this.isActive ? '0' : null;
  }

  @HostBinding('attr.title')
  get title(): string | null {
    const term = findTeachingTerm(this.termId);
    return this.isActive && term ? `查看教学文档：${term.title}` : null;
  }

  @HostListener('click', ['$event'])
  onClick(event: MouseEvent): void {
    if (!this.isActive || !this.termId) return;
    event.preventDefault();
    event.stopPropagation();
    this.teachingSearch.setActive(false);
    this.openTeachingDoc();
  }

  @HostListener('keydown.enter', ['$event'])
  onEnter(event: Event): void {
    this.openFromKeyboard(event);
  }

  @HostListener('keydown.space', ['$event'])
  onSpace(event: Event): void {
    this.openFromKeyboard(event);
  }

  private openFromKeyboard(event: Event): void {
    if (!this.isActive || !this.termId) return;
    event.preventDefault();
    this.teachingSearch.setActive(false);
    this.openTeachingDoc();
  }

  private openTeachingDoc(): void {
    const url = `/teaching#${encodeURIComponent(this.termId)}`;
    const target = window.open(url, TEACHING_WINDOW_NAME);
    target?.focus();
  }
}
