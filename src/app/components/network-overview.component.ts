import { CommonModule } from '@angular/common';
import { Component, EventEmitter, Input, Output } from '@angular/core';
import { Connection, NetworkLayer } from '../sim-models';

interface DrawNode {
  layerIndex: number;
  nodeIndex: number;
  x: number;
  y: number;
}

interface DrawLayer {
  index: number;
  id: number;
  x: number;
  yTop: number;
  yBottom: number;
  displayNodes: number;
  units: number;
  type: string;
  nodes: DrawNode[];
}

interface NodeSelectionEvent {
  layerId: number;
  nodeIndex: number;
  append: boolean;
}

@Component({
  selector: 'app-network-overview',
  standalone: true,
  imports: [CommonModule],
  template: `
    <div class="network-overview">
      <svg
        [attr.viewBox]="'0 0 ' + svgWidth + ' 420'"
        preserveAspectRatio="none"
        role="img"
        aria-label="network overview"
      >
        @for (pair of layerPairs; track $index) {
          @for (fromNode of pair.from.nodes; track $index) {
            @for (toNode of pair.to.nodes; track $index) {
              <path
                [attr.d]="linkPath(fromNode, toNode)"
                [attr.stroke]="linkColor(pair.from.index, fromNode.nodeIndex, pair.to.index, toNode.nodeIndex)"
                [attr.stroke-width]="linkWidth(pair.from.index, fromNode.nodeIndex, pair.to.index, toNode.nodeIndex)"
                class="link"
              ></path>
            }
          }
        }

        @for (layer of drawLayers; track layer.id) {
          <g class="layer-title" (click)="layerSelected.emit(layer.id)">
            <text [attr.x]="layer.x" [attr.y]="24" text-anchor="middle">{{ layer.type }}</text>
            <text [attr.x]="layer.x" [attr.y]="40" text-anchor="middle" class="units">{{ layer.units }} units</text>
          </g>

          @if (layer.index > 0 && layer.index < drawLayers.length - 1) {
            <g class="layer-controls">
              <rect [attr.x]="layer.x - 24" y="48" width="18" height="18" rx="4" (click)="neuronDelta.emit({ layerId: layer.id, delta: 1 })"></rect>
              <text [attr.x]="layer.x - 15" y="61" text-anchor="middle" (click)="neuronDelta.emit({ layerId: layer.id, delta: 1 })">+</text>

              <rect [attr.x]="layer.x + 6" y="48" width="18" height="18" rx="4" (click)="neuronDelta.emit({ layerId: layer.id, delta: -1 })"></rect>
              <text [attr.x]="layer.x + 15" y="61" text-anchor="middle" (click)="neuronDelta.emit({ layerId: layer.id, delta: -1 })">-</text>
            </g>
          }

          @for (node of layer.nodes; track $index) {
            <g (click)="onNodeClick(layer.id, node.nodeIndex, $event)" class="node-group">
              <circle
                [attr.cx]="node.x"
                [attr.cy]="node.y"
                r="13"
                [class.active]="layer.id === selectedLayerId"
                [class.node-picked]="isNodeSelected(layer.id, node.nodeIndex)"
              ></circle>
            </g>
          }

          @if (layer.units > layer.displayNodes) {
            <text [attr.x]="layer.x" [attr.y]="layer.yBottom + 18" class="ellipsis" text-anchor="middle">...</text>
          }
        }
      </svg>
    </div>
  `,
  styles: [
    `
      .network-overview {
        border: 1px solid #d6deea;
        border-radius: 10px;
        background: linear-gradient(180deg, #f8fbff 0%, #f0f6fd 100%);
        padding: 8px;
      }

      svg {
        width: 100%;
        height: 360px;
        min-width: 860px;
      }

      .layer-title text {
        fill: #334155;
        font-size: 12px;
        cursor: pointer;
      }

      .layer-title .units {
        fill: #64748b;
        font-size: 10px;
      }

      .link {
        fill: none;
        stroke-linecap: round;
        opacity: 0.9;
      }

      .layer-controls rect {
        fill: #ffffff;
        stroke: #b8c6d8;
        cursor: pointer;
      }

      .layer-controls text {
        fill: #1f2937;
        font-size: 12px;
        font-weight: 700;
        cursor: pointer;
      }

      circle {
        fill: #ffffff;
        stroke: #8ea2ba;
        stroke-width: 1.8;
        cursor: pointer;
      }

      circle.active {
        fill: #dbeafe;
        stroke: #2563eb;
      }

      circle.node-picked {
        fill: #fde68a;
        stroke: #d97706;
        stroke-width: 2.2;
      }

      text {
        font-size: 11px;
        fill: #1f2937;
        pointer-events: none;
      }

      .ellipsis {
        fill: #64748b;
        font-size: 16px;
      }
    `
  ]
})
export class NetworkOverviewComponent {
  @Input() layers: NetworkLayer[] = [];
  @Input() connections: Connection[] = [];
  @Input() selectedLayerId = -1;
  @Input() selectedNodeKeys: string[] = [];
  @Input() zoom = 1;
  @Output() layerSelected = new EventEmitter<number>();
  @Output() neuronDelta = new EventEmitter<{ layerId: number; delta: number }>();
  @Output() nodeSelected = new EventEmitter<NodeSelectionEvent>();

  get svgWidth(): number {
    const base = 980;
    const extra = Math.max(0, this.layers.length - 6) * 180;
    return Math.round((base + extra) * Math.max(1, this.zoom));
  }

  get drawLayers(): DrawLayer[] {
    const count = Math.max(1, this.layers.length);
    const yTop = 86;
    const yBottom = 310;

    return this.layers.map((layer, idx) => {
      const x = 50 + (idx * (this.svgWidth - 100)) / Math.max(1, count - 1);
      const displayNodes = this.displayNodeCount(layer);
      const spacing = displayNodes > 1 ? (yBottom - yTop) / (displayNodes - 1) : 0;
      const nodes: DrawNode[] = Array.from({ length: displayNodes }, (_, nodeIdx) => {
        return {
          layerIndex: idx,
          nodeIndex: nodeIdx,
          x,
          y: yTop + spacing * nodeIdx
        };
      });

      return {
        index: idx,
        id: layer.id,
        x,
        yTop,
        yBottom,
        displayNodes,
        units: layer.units,
        type: layer.type,
        nodes
      };
    });
  }

  get layerPairs(): Array<{ from: DrawLayer; to: DrawLayer }> {
    const pairs: Array<{ from: DrawLayer; to: DrawLayer }> = [];
    for (let i = 0; i < this.drawLayers.length - 1; i += 1) {
      pairs.push({ from: this.drawLayers[i], to: this.drawLayers[i + 1] });
    }
    return pairs;
  }

  linkPath(fromNode: DrawNode, toNode: DrawNode): string {
    const mid = (fromNode.x + toNode.x) / 2;
    return `M${fromNode.x},${fromNode.y} C${mid},${fromNode.y} ${mid},${toNode.y} ${toNode.x},${toNode.y}`;
  }

  onNodeClick(layerId: number, nodeIndex: number, event: MouseEvent): void {
    event.stopPropagation();
    this.layerSelected.emit(layerId);
    this.nodeSelected.emit({
      layerId,
      nodeIndex,
      append: event.ctrlKey || event.metaKey || event.shiftKey
    });
  }

  isNodeSelected(layerId: number, nodeIndex: number): boolean {
    return this.selectedNodeKeys.includes(`${layerId}-${nodeIndex}`);
  }

  linkColor(fromLayer: number, fromNode: number, toLayer: number, toNode: number): string {
    const w = this.syntheticWeight(fromLayer, fromNode, toLayer, toNode);
    const intensity = Math.min(1, Math.abs(w));
    if (w >= 0) {
      const r = Math.round(225 - intensity * 130);
      const g = Math.round(235 - intensity * 70);
      const b = Math.round(245);
      return `rgb(${r},${g},${b})`;
    }
    const r = Math.round(245);
    const g = Math.round(228 - intensity * 85);
    const b = Math.round(210 - intensity * 120);
    return `rgb(${r},${g},${b})`;
  }

  linkWidth(fromLayer: number, fromNode: number, toLayer: number, toNode: number): number {
    const w = Math.abs(this.syntheticWeight(fromLayer, fromNode, toLayer, toNode));
    return 0.6 + w * 2.2;
  }

  private displayNodeCount(layer: NetworkLayer): number {
    if (layer.type === 'input') {
      return 7;
    }
    if (layer.type === 'output') {
      return Math.min(8, Math.max(2, layer.units));
    }

    const scaled = Math.round(Math.sqrt(Math.max(1, layer.units)));
    return Math.min(8, Math.max(3, scaled));
  }

  private syntheticWeight(fromLayer: number, fromNode: number, toLayer: number, toNode: number): number {
    const seed = Math.sin((fromLayer + 1) * 12.9898 + (fromNode + 1) * 78.233 + (toLayer + 1) * 37.719 + (toNode + 1) * 11.131) * 43758.5453;
    const frac = seed - Math.floor(seed);
    return frac * 2 - 1;
  }
}
