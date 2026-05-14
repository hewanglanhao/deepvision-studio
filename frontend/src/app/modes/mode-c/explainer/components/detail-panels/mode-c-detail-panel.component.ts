import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { TeachingTermDirective } from '@shared/teaching/teaching-term.directive';
import { ModeCNetworkLayer } from '../../models/mode-c.types';
import { ModeCModelService } from '../../services/mode-c-model.service';
import { ModeCStateService } from '../../services/mode-c-state.service';

@Component({
  selector: 'app-mode-c-detail-panel',
  imports: [CommonModule, TeachingTermDirective],
  templateUrl: './mode-c-detail-panel.component.html',
  styleUrl: './mode-c-detail-panel.component.css'
})
export class ModeCDetailPanelComponent {
  constructor(
    readonly model: ModeCModelService,
    readonly state: ModeCStateService
  ) {}

  selectTopic(topicId: string): void {
    this.state.setSelectedTopic(topicId);
  }

  isConvLayer(layer: ModeCNetworkLayer): boolean {
    return layer.type === 'conv';
  }

  isReluLayer(layer: ModeCNetworkLayer): boolean {
    return layer.type === 'relu';
  }

  isPoolLayer(layer: ModeCNetworkLayer): boolean {
    return layer.type === 'pool';
  }

  isFlattenLayer(layer: ModeCNetworkLayer): boolean {
    return layer.type === 'flatten';
  }

  isOutputLayer(layer: ModeCNetworkLayer): boolean {
    return layer.type === 'output';
  }

  isInputLayer(layer: ModeCNetworkLayer): boolean {
    return layer.type === 'input';
  }

  formatPercent(value: number): string {
    return `${(value * 100).toFixed(1)}%`;
  }

  formatSigned(value: number, digits = 4): string {
    const formatted = value.toFixed(digits);
    return value > 0 ? `+${formatted}` : formatted;
  }

  getLayerNarrative(layer: ModeCNetworkLayer): string {
    if (layer.type === 'input') {
      return '这里展示的是经过中心裁剪和缩放后，真正送入网络的标准化输入图像。';
    }
    if (layer.type === 'conv') {
      return '卷积层会用学习得到的卷积核去扫描局部区域。这里的预览图强调的是卷积核响应最强的位置。';
    }
    if (layer.type === 'relu') {
      return 'ReLU 会去掉负响应。这里的正值占比能帮助判断激活后的特征图到底有多稀疏。';
    }
    if (layer.type === 'pool') {
      return '池化会压缩空间分辨率，同时尽量保留最强响应区域。预览会更粗糙，但通常仍会保留前一层中的高能区域。';
    }
    if (layer.type === 'flatten') {
      return 'Flatten 会把最后的特征堆栈拉平成一维向量，供分类层使用。这里的预览是这个向量的紧凑网格投影。';
    }
    return '输出层会把拉平后的特征向量映射成各类别的 logits，下面的排名来自当前样例的真实 softmax 结果。';
  }

  selectChannel(index: number): void {
    this.state.setSelectedChannel(index);
  }

  getActiveChannelPreview() {
    const previews = this.state.selectedLayerDetail()?.channelPreviews ?? [];
    if (!previews.length) return null;
    return previews.find(preview => preview.index === this.state.selectedChannelIndex()) ?? previews[0];
  }

  getVisibleVectorValues(): Array<{ index: number; value: number }> {
    const values = this.state.selectedLayerDetail()?.vectorValues ?? [];
    return values.map((value, index) => ({ index, value }));
  }

  getVectorMagnitude(value: number): number {
    return Math.max(8, Math.min(100, Math.abs(value) * 100));
  }

  getActiveConvExample() {
    const examples = this.state.selectedLayerDetail()?.convExamples ?? [];
    if (!examples.length) return null;
    return examples.find(example => example.outputChannelIndex === this.state.selectedChannelIndex()) ?? examples[0];
  }

  getActiveReluExample() {
    const examples = this.state.selectedLayerDetail()?.reluExamples ?? [];
    if (!examples.length) return null;
    return examples.find(example => example.channelIndex === this.state.selectedChannelIndex()) ?? examples[0];
  }

  getActivePoolExample() {
    const examples = this.state.selectedLayerDetail()?.poolExamples ?? [];
    if (!examples.length) return null;
    return examples.find(example => example.channelIndex === this.state.selectedChannelIndex()) ?? examples[0];
  }

  getPredictionSummaryRows(limit = 3) {
    return (this.state.currentSamplePrediction()?.topClasses ?? []).slice(0, limit);
  }

  getProbabilityRows() {
    return this.state.currentSamplePrediction()?.topClasses ?? [];
  }

  getPredictionRank(label: string): number | null {
    const index = this.getProbabilityRows().findIndex(candidate => candidate.label === label);
    return index >= 0 ? index + 1 : null;
  }
}
