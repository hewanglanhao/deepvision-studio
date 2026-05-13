import { CommonModule, ViewportScroller } from '@angular/common';
import { Component, OnInit } from '@angular/core';
import { ActivatedRoute, RouterLink } from '@angular/router';
import { TEACHING_TERMS, TeachingTerm } from '@shared/teaching/teaching-glossary';

@Component({
  selector: 'app-teaching-doc-page',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './teaching-doc-page.component.html',
  styleUrl: './teaching-doc-page.component.css'
})
export class TeachingDocPageComponent implements OnInit {
  readonly terms = TEACHING_TERMS;
  readonly categories: TeachingTerm['category'][] = Array.from(
    new Set(this.terms.map(term => term.category))
  );

  constructor(
    private readonly route: ActivatedRoute,
    private readonly viewportScroller: ViewportScroller
  ) {}

  ngOnInit(): void {
    this.viewportScroller.setOffset([0, 18]);
    this.route.fragment.subscribe(fragment => {
      if (!fragment) return;
      window.setTimeout(() => this.viewportScroller.scrollToAnchor(fragment), 0);
    });
  }

  termsByCategory(category: TeachingTerm['category']): TeachingTerm[] {
    return this.terms.filter(term => term.category === category);
  }
}
