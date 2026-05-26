import { CommonModule } from '@angular/common';
import { Component } from '@angular/core';
import { ModeDVectorBar } from '../../models/mode-d.types';
import { ModeDStateService } from '../../services/mode-d-state.service';

type QkvStage = 'query' | 'key' | 'score' | 'value' | 'output';

@Component({
  selector: 'app-mode-d-qkv-panel',
  imports: [CommonModule],
  templateUrl: './mode-d-qkv-panel.component.html',
  styleUrl: './mode-d-qkv-panel.component.css'
})
export class ModeDQkvPanelComponent {
  hoveredStage: QkvStage | null = null;
  lockedStage: QkvStage | null = null;
  hoveredDimensionIndex: number | null = null;

  constructor(readonly state: ModeDStateService) {}

  focusModeLabel(mode: 'selected' | 'hovered' | 'strongest'): string {
    switch (mode) {
      case 'selected':
        return '已锁定';
      case 'hovered':
        return '悬停中';
      default:
        return '自动聚焦';
    }
  }

  attentionPercent(): string {
    return `${(this.state.qkvTeaching().attentionWeight * 100).toFixed(1)}%`;
  }

  querySummary(): string {
    const detail = this.state.activeAttentionDetail();
    return `把“${detail.sourceToken}”看成当前 token 发出的检索请求：它正在寻找哪些上下文最能帮助自己更新表示。`;
  }

  keySummary(): string {
    const detail = this.state.activeAttentionDetail();
    return `把“${detail.targetToken}”看成候选键：它之所以会被关注，是因为它的 Key 与当前 Query 更匹配。`;
  }

  valueSummary(): string {
    const detail = this.state.activeAttentionDetail();
    return `当“${detail.targetToken}”被选中后，它对应的 Value 会按 ${(detail.weight * 100).toFixed(1)}% 的权重汇入当前 token 的新表示。`;
  }

  dotProduct(): number {
    const { queryVector, keyVector } = this.state.qkvTeaching();
    return queryVector.reduce((sum, item, index) => sum + item.value * (keyVector[index]?.value ?? 0), 0);
  }

  scaledScore(): number {
    const dimension = this.state.qkvTeaching().queryVector.length || 1;
    return this.dotProduct() / Math.sqrt(dimension);
  }

  weightedValue(vector: ModeDVectorBar): number {
    return vector.value * this.state.qkvTeaching().attentionWeight;
  }

  barWidth(value: number): string {
    return `${Math.min(100, Math.max(10, Math.abs(value) * 100))}%`;
  }

  barTrackClass(value: number): string {
    return value >= 0 ? 'positive' : 'negative';
  }

  activeStage(): QkvStage | null {
    return this.lockedStage ?? this.hoveredStage;
  }

  stageTitle(stage: QkvStage): string {
    switch (stage) {
      case 'query':
        return 'Query：发起检索';
      case 'key':
        return 'Key：判断是否匹配';
      case 'score':
        return '得分：把匹配程度变成注意力权重';
      case 'value':
        return 'Value：准备被汇入输出的信息';
      case 'output':
        return '输出：把加权后的 Value 融入当前 token';
    }
  }

  stageExplanation(stage: QkvStage): string {
    const detail = this.state.activeAttentionDetail();
    switch (stage) {
      case 'query':
        return `当前由“${detail.sourceToken}”发起检索，它决定模型此刻想从上下文里找回什么信息。`;
      case 'key':
        return `“${detail.targetToken}”提供可被匹配的键。如果它和 Query 越契合，这条连接就越容易被放大。`;
      case 'score':
        return '点积和缩放先把 Query 与 Key 的匹配程度变成分数，softmax 再把它变成最终分给这个 Value 的注意力权重。';
      case 'value':
        return `一旦“${detail.targetToken}”被选中，它携带的 Value 就会按当前权重参与输出更新。`;
      case 'output':
        return `最终输出不是简单复制某个 token，而是把多个 Value 按权重混合后，回流到“${detail.sourceToken}”的新表示里。`;
    }
  }

  stageMiniTitle(stage: 'query' | 'key' | 'value'): string {
    switch (stage) {
      case 'query':
        return '这个 Query 在问什么';
      case 'key':
        return '这个 Key 为什么会被匹配';
      case 'value':
        return '这个 Value 会带回什么';
    }
  }

  stageMiniCopy(stage: 'query' | 'key' | 'value'): string {
    switch (stage) {
      case 'query':
        return '它代表当前 token 的检索意图，决定模型此刻想从上下文里寻找哪类信息。';
      case 'key':
        return '它代表候选上下文的可匹配线索，决定自己会不会被当前 Query 选中。';
      case 'value':
        return '它代表真正被传回输出的内容，注意力权重越高，这部分内容的影响就越大。';
    }
  }

  scoreMiniTitle(): string {
    return '这个得分为什么重要';
  }

  scoreMiniCopy(): string {
    return '它把 Query 与 Key 的匹配程度压缩成一个可比较的分数，再经过 softmax 变成真正分配给 Value 的注意力权重。';
  }

  currentDimensionLabel(): string {
    return this.hoveredDimensionIndex === null ? '未聚焦' : `d${this.hoveredDimensionIndex + 1}`;
  }

  currentDimensionHint(): string {
    if (this.hoveredDimensionIndex === null) {
      return '悬停任一向量分量后，会同步高亮 Query、Key、Value 中对应的同一维度。';
    }

    return `当前正在比较第 ${this.hoveredDimensionIndex + 1} 维，它会同时影响 Query 与 Key 的点积，并决定这部分 Value 如何参与输出。`;
  }

  hoverStage(stage: QkvStage): void {
    if (!this.lockedStage) {
      this.hoveredStage = stage;
    }
  }

  leaveStage(stage: QkvStage): void {
    if (!this.lockedStage && this.hoveredStage === stage) {
      this.hoveredStage = null;
    }
  }

  toggleStage(stage: QkvStage): void {
    this.lockedStage = this.lockedStage === stage ? null : stage;
  }

  clearLockedStage(): void {
    this.lockedStage = null;
  }

  hoverDimension(index: number): void {
    this.hoveredDimensionIndex = index;
  }

  clearHoveredDimension(): void {
    this.hoveredDimensionIndex = null;
  }

  isDimensionHighlighted(index: number): boolean {
    return this.hoveredDimensionIndex === index;
  }

  isStageFocused(stage: QkvStage): boolean {
    return this.activeStage() === stage;
  }

  isStageDimmed(stages: QkvStage[]): boolean {
    const active = this.activeStage();
    return !!active && !stages.includes(active);
  }
}
