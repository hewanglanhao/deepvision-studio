import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ModeDStateService } from '../../services/mode-d-state.service';

@Component({
  selector: 'app-mode-d-input-panel',
  imports: [CommonModule, FormsModule],
  templateUrl: './mode-d-input-panel.component.html',
  styleUrl: './mode-d-input-panel.component.css'
})
export class ModeDInputPanelComponent {
  constructor(readonly state: ModeDStateService) {}
}

