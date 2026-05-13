import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ModeDAssetsService } from '../../services/mode-d-assets.service';

@Component({
  selector: 'app-mode-d-article',
  imports: [CommonModule],
  templateUrl: './mode-d-article.component.html',
  styleUrl: './mode-d-article.component.css',
})
export class ModeDArticleComponent {
  constructor(readonly assets: ModeDAssetsService) {}
}
