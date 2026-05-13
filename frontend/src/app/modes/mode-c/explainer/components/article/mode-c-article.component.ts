import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { ModeCStateService } from '../../services/mode-c-state.service';

@Component({
  selector: 'app-mode-c-article',
  imports: [CommonModule],
  templateUrl: './mode-c-article.component.html',
  styleUrl: './mode-c-article.component.css'
})
export class ModeCArticleComponent {
  constructor(readonly state: ModeCStateService) {}
}
