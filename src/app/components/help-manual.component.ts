import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';

@Component({
  selector: 'app-help-manual',
  standalone: true,
  imports: [CommonModule],
  template: `
    @if (open) {
      <div class="help-mask" (click)="closed.emit()">
        <section class="help-card" (click)="$event.stopPropagation()">
          <header>
            <h3>新手帮助手册</h3>
            <button (click)="closed.emit()">关闭</button>
          </header>

          <div class="help-content">
            <p>这是一个教学型深度学习可视化平台，建议按以下顺序体验：</p>
            <ol>
              <li>在顶部选择模型模板（MLP/CNN/ResNet Mini）。</li>
              <li>在 Network 区拖拽层并调整参数（卷积核、激活函数、Dropout）。</li>
              <li>点击开始训练，观察 Loss、Train Acc、Val Acc 曲线。</li>
              <li>在 Data 区点击样本，查看单样本前向传播激活条形图。</li>
              <li>在 Output 区查看特征图、Grad-CAM、混淆矩阵、loss 散点与谷地图。</li>
              <li>使用“真实图片卷积实验”上传图片，观察卷积与池化结果。</li>
            </ol>

            <h4>常见问题</h4>
            <p><b>Q:</b> 为什么是模拟训练？</p>
            <p><b>A:</b> 当前是前端演示版，后续可接 Spring Boot 接口替换为真实训练数据。</p>

            <p><b>Q:</b> 看不懂曲线？</p>
            <p><b>A:</b> 通常期望 Loss 下降，Train/Val Accuracy 上升且差距不要过大。</p>
          </div>
        </section>
      </div>
    }
  `,
  styles: [
    `
      .help-mask {
        position: fixed;
        inset: 0;
        background: rgba(2, 6, 23, 0.55);
        display: grid;
        place-items: center;
        z-index: 1200;
      }

      .help-card {
        width: min(820px, calc(100vw - 20px));
        max-height: min(86vh, 760px);
        overflow: auto;
        background: #ffffff;
        border-radius: 12px;
        border: 1px solid #d4dbe5;
        box-shadow: 0 18px 40px rgba(15, 23, 42, 0.28);
      }

      header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 12px 14px;
        border-bottom: 1px solid #e2e8f0;
        position: sticky;
        top: 0;
        background: #f8fafc;
      }

      h3 {
        margin: 0;
        font-size: 18px;
      }

      button {
        border: 1px solid #c8d4e3;
        background: #fff;
        border-radius: 8px;
        padding: 6px 10px;
        cursor: pointer;
      }

      .help-content {
        padding: 14px;
        color: #334155;
        line-height: 1.6;
      }

      ol {
        margin: 8px 0 12px 18px;
      }

      h4 {
        margin: 10px 0 4px;
      }

      p {
        margin: 4px 0;
      }
    `
  ]
})
export class HelpManualComponent {
  @Input() open = false;
  @Output() closed = new EventEmitter<void>();
}
