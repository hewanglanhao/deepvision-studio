import { Injectable, computed, signal } from '@angular/core';
import { ModeCModelStatusSummary } from '../models/mode-c.types';

@Injectable({ providedIn: 'root' })
export class ModeCModelService {
  private readonly initialized = signal(false);

  readonly shellStatus = computed<ModeCModelStatusSummary>(() => {
    if (this.initialized()) {
      return {
        title: 'Native shell ready',
        description: 'The Angular structure is in place and ready for real model loading work in the next phase.',
        status: 'ready'
      };
    }

    return {
      title: 'Bootstrapping',
      description: 'Preparing the Mode C Angular shell.',
      status: 'in-progress'
    };
  });

  initializeNativeShell(): void {
    if (this.initialized()) return;
    this.initialized.set(true);
  }
}
