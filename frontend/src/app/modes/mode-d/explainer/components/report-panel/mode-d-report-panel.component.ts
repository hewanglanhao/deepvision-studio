import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { ModeDStateService } from '../../services/mode-d-state.service';

@Component({
  selector: 'app-mode-d-report-panel',
  imports: [CommonModule],
  templateUrl: './mode-d-report-panel.component.html',
  styleUrl: './mode-d-report-panel.component.css'
})
export class ModeDReportPanelComponent {
  constructor(readonly state: ModeDStateService) {}
}

