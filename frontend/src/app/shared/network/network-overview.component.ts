import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { NetworkLayer } from '@shared/simulation/sim-models';

const LAYER_COLOR: Record<string, string> = {
  input: '#6366f1', conv2d: '#0ea5e9', pool2d: '#10b981',
  residual: '#14b8a6', flatten: '#f59e0b', dense: '#8b5cf6', activation: '#ec4899',
  dropout: '#94a3b8', output: '#ef4444'
};
const LAYER_ICON: Record<string, string> = {
  input: '⬛', conv2d: '⊞', pool2d: '⊟', flatten: '≡',
  residual: '+', dense: '◉', activation: 'ƒ', dropout: '⊘', output: '▶'
};

@Component({
  selector: 'app-network-overview',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="nw-scroll" (dragover)="onDragOver($event)" (drop)="onDropCanvas($event)">
      <div class="nw-track">
        @for (layer of layers; track layer.id; let i = $index) {
          <!-- 插入区 -->
          <div class="drop-slot"
               [class.drag-over]="dropTargetIndex === i"
               (dragover)="onSlotOver($event, i)"
               (dragleave)="dropTargetIndex = -1"
               (drop)="onDropSlot($event, i)">
            <div class="slot-line"></div>
          </div>

          <!-- 层卡片 -->
          <div class="layer-card"
               [class.selected]="layer.id === selectedLayerId"
               [class.has-error]="hasError(layer.id)"
               [class.dragging]="dragSourceIndex === i"
               [style.--accent]="color(layer.type)"
               draggable="true"
               (dragstart)="onCardDragStart($event, i)"
               (dragend)="onCardDragEnd()"
               (click)="layerSelected.emit(layer.id)">
            <div class="card-top-bar" [style.background]="color(layer.type)"></div>
            <div class="card-icon" [style.color]="color(layer.type)">{{ icon(layer.type) }}</div>
            <div class="card-name">{{ layer.name }}</div>
            <div class="card-type" [style.color]="color(layer.type)">{{ typeLabel(layer.type) }}</div>
            @if (shapeHints[layer.id]) {
              <div class="card-shape">{{ shapeHints[layer.id] }}</div>
            }
            @if (layer.type === 'conv2d') {
              <div class="card-badge">{{ layer.params.outChannels }}ch · k{{ layer.params.kernelSize }}</div>
            }
            @if (layer.type === 'dense' || layer.type === 'output') {
              <div class="card-badge">{{ layer.params.units }} 单元</div>
            }
            @if (layer.type === 'pool2d') {
              <div class="card-badge">{{ layer.params.mode }} · k{{ layer.params.kernelSize }}</div>
            }
            @if (layer.type === 'residual') {
              <div class="card-badge">{{ layer.params.outChannels }}ch · {{ layer.params.useProjection ? 'proj' : 'skip' }}</div>
            }
          </div>

          <!-- 连接箭头 -->
          @if (i < layers.length - 1) {
            <div class="nw-arrow">
              <svg width="32" height="24" viewBox="0 0 32 24">
                <line x1="0" y1="12" x2="24" y2="12" stroke="#334155" stroke-width="1.5"/>
                <polygon points="24,7 32,12 24,17" fill="#334155"/>
              </svg>
            </div>
          }
        }

        <!-- 末尾插入区 -->
        <div class="drop-slot"
             [class.drag-over]="dropTargetIndex === layers.length"
             (dragover)="onSlotOver($event, layers.length)"
             (dragleave)="dropTargetIndex = -1"
             (drop)="onDropSlot($event, layers.length)">
          <div class="slot-line"></div>
        </div>
      </div>
    </div>
  `,
  styles: [`
    .nw-scroll {
      overflow-x: auto; overflow-y: hidden;
      padding: 10px 8px; min-height: 118px;
    }
    .nw-scroll::-webkit-scrollbar { height: 5px; }
    .nw-scroll::-webkit-scrollbar-thumb { background: #d0d7de; border-radius: 999px; }

    .nw-track {
      display: flex; align-items: center;
      gap: 0; min-width: max-content;
    }

    .layer-card {
      width: 106px; min-height: 98px;
      background: #fff;
      border: 1.5px solid #e2e6ea;
      border-radius: 12px;
      padding: 7px 8px 6px;
      cursor: pointer; flex-shrink: 0;
      transition: border-color .15s, box-shadow .15s, transform .1s;
      position: relative; overflow: hidden;
      display: flex; flex-direction: column; align-items: center; gap: 3px;
      box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    .layer-card:hover { border-color: var(--accent,#2563eb); box-shadow: 0 2px 8px rgba(37,99,235,.12); }
    .layer-card.selected {
      border-color: var(--accent,#2563eb);
      box-shadow: 0 0 0 2px rgba(37,99,235,.2), 0 2px 12px rgba(37,99,235,.15);
      background: #eff6ff;
    }
    .layer-card.has-error { border-color: #dc2626 !important; background: #fef2f2 !important; }
    .layer-card.has-error .card-name { color: #991b1b; }
    .layer-card.dragging { opacity: .35; transform: scale(.95); }

    .card-top-bar {
      position: absolute; top: 0; left: 0; right: 0;
      height: 3px; border-radius: 12px 12px 0 0;
    }
    .card-icon { font-size: 20px; margin-top: 5px; line-height: 1; }
    .card-name { font-size: 11px; font-weight: 600; color: #1a2332; text-align: center; line-height: 1.3; }
    .card-type { font-size: 10px; font-weight: 700; text-align: center; letter-spacing: .04em; }
    .card-shape { font-size: 9px; font-family: monospace; color: #8a9ab0; text-align: center; }
    .card-badge {
      font-size: 9px; color: #8a9ab0; text-align: center;
      background: #f7f8fa; border-radius: 999px; padding: 1px 6px;
      border: 1px solid #e2e6ea;
    }

    .nw-arrow { display: flex; align-items: center; flex-shrink: 0; }

    .drop-slot {
      width: 16px; min-height: 98px; flex-shrink: 0;
      display: flex; align-items: center; justify-content: center;
      transition: width .15s;
    }
    .drop-slot.drag-over { width: 44px; }
    .slot-line {
      width: 2px; height: 50px; background: transparent;
      border-radius: 999px; transition: all .15s;
    }
    .drop-slot.drag-over .slot-line { background: #2563eb; box-shadow: 0 0 6px rgba(37,99,235,.4); }
  `]
})
export class NetworkOverviewComponent {
  @Input() layers: NetworkLayer[] = [];
  @Input() selectedLayerId = -1;
  @Input() shapeHints: Record<number, string> = {};
  @Input() errorLayerIds: number[] = [];
  @Output() layerSelected = new EventEmitter<number>();
  @Output() layersReordered = new EventEmitter<NetworkLayer[]>();
  @Output() newLayerDropped = new EventEmitter<{ type: string; index: number }>();

  hasError(id: number): boolean { return this.errorLayerIds.includes(id); }

  dragSourceIndex = -1;
  dropTargetIndex = -1;
  private dragType: 'card' | 'new' = 'card';
  private dragNewType = '';

  color(t: string) { return LAYER_COLOR[t] ?? '#64748b'; }
  icon(t: string)  { return LAYER_ICON[t] ?? '□'; }

  typeLabel(type: string): string {
    const m: Record<string, string> = {
      input: 'Input', conv2d: 'Conv2D', pool2d: 'Pool2D', flatten: 'Flatten',
      residual: 'Residual', dense: 'Dense', activation: 'Activation', dropout: 'Dropout', output: 'Output'
    };
    return m[type] ?? type;
  }

  onCardDragStart(e: DragEvent, i: number): void {
    this.dragSourceIndex = i;
    this.dragType = 'card';
    e.dataTransfer?.setData('text/plain', String(i));
  }
  onCardDragEnd(): void { this.dragSourceIndex = -1; this.dropTargetIndex = -1; }

  onSlotOver(e: DragEvent, i: number): void { e.preventDefault(); this.dropTargetIndex = i; }
  onDragOver(e: DragEvent): void { e.preventDefault(); }

  onDropSlot(e: DragEvent, targetIndex: number): void {
    e.preventDefault();
    this.dropTargetIndex = -1;
    const data = e.dataTransfer?.getData('text/plain') ?? '';

    if (this.dragType === 'new') {
      this.newLayerDropped.emit({ type: this.dragNewType, index: targetIndex });
      return;
    }

    const from = this.dragSourceIndex;
    if (from < 0 || from === targetIndex || from === targetIndex - 1) return;

    const arr = [...this.layers];
    const [moved] = arr.splice(from, 1);
    const insertAt = targetIndex > from ? targetIndex - 1 : targetIndex;
    arr.splice(insertAt, 0, moved);
    this.layersReordered.emit(arr);
    this.dragSourceIndex = -1;
  }

  onDropCanvas(e: DragEvent): void { e.preventDefault(); }

  /** 供外部 palette chip 调用：设置拖拽类型为 new layer */
  setDragNew(type: string): void { this.dragType = 'new'; this.dragNewType = type; }
}
