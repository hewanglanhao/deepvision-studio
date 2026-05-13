import { Injectable, signal } from '@angular/core';

@Injectable({ providedIn: 'root' })
export class TeachingSearchService {
  readonly active = signal(false);

  setActive(value: boolean): void {
    this.active.set(value);
  }

  toggle(): void {
    this.active.update(value => !value);
  }
}
