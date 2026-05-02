import { Component } from '@angular/core';
import { DomSanitizer, SafeResourceUrl } from '@angular/platform-browser';
import { RouterLink } from '@angular/router';

@Component({
  selector: 'app-mode-c-page',
  imports: [RouterLink],
  templateUrl: './mode-c-page.component.html',
  styleUrl: './mode-c-page.component.css'
})
export class ModeCPageComponent {
  readonly moduleUrl: SafeResourceUrl;
  readonly assetTarget = '/modules/cnn-explainer/index.html';
  readonly sourceProjectPath = 'D:\\VS Code\\cnn-explainer';
  readonly syncCommand = '.\\scripts\\sync-cnn-explainer.ps1';

  constructor(private sanitizer: DomSanitizer) {
    this.moduleUrl = this.sanitizer.bypassSecurityTrustResourceUrl(this.assetTarget);
  }
}
