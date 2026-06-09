import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModeEAssetsService } from '../../services/mode-e-assets.service';

@Component({
  selector: 'app-mode-e-article',
  imports: [CommonModule],
  templateUrl: './mode-e-article.component.html',
  styleUrl: './mode-e-article.component.css',
})
export class ModeEArticleComponent {
  constructor(readonly assets: ModeEAssetsService) {}
}
