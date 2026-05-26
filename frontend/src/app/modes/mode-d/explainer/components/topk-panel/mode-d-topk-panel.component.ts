import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { ModeDStateService } from '../../services/mode-d-state.service';

@Component({
  selector: 'app-mode-d-topk-panel',
  imports: [CommonModule],
  templateUrl: './mode-d-topk-panel.component.html',
  styleUrl: './mode-d-topk-panel.component.css'
})
export class ModeDTopKPanelComponent {
  constructor(readonly state: ModeDStateService) {}
}

