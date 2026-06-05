import { NetworkLayer, TensorShape } from '@shared/simulation/sim-models';
import { Network3dLayerView } from './network-3d.models';

const LAYER_COLORS: Record<string, string> = {
  input: '#6366f1',
  conv2d: '#0ea5e9',
  pool2d: '#10b981',
  residual: '#14b8a6',
  flatten: '#f59e0b',
  dense: '#8b5cf6',
  activation: '#ec4899',
  dropout: '#94a3b8',
  output: '#ef4444'
};

/** 把 A 模式网络层转换成 3D 布局节点；卷积/池化层按特征图尺寸绘制，Dense/Output 按神经元数量绘制。 */
export function buildNetwork3dLayerViews(
  layers: NetworkLayer[],
  layerShapes: Record<number, TensorShape>,
  shapeHints: Record<number, string>
): Network3dLayerView[] {
  return layers.map((layer) => {
    const shape = layerShapes[layer.id] ?? [];
    const size = layerSize(shape, layer);
    return {
      layer,
      shape,
      shapeLabel: shapeHints[layer.id] || formatShape(shape),
      width: size.width,
      height: size.height,
      depth: size.depth,
      color: LAYER_COLORS[layer.type] ?? '#64748b'
    };
  });
}

/** 格式化张量 shape，方便在 3D 面板中查看每层特征图或向量长度。 */
export function formatShape(shape: TensorShape): string {
  return shape.length ? `[${shape.join(', ')}]` : '[]';
}

function layerSize(shape: TensorShape, layer: NetworkLayer): { width: number; height: number; depth: number } {
  if (shape.length === 3) {
    const [height, width, channels] = shape;
    return {
      width: clamp(width / 24, 0.9, 3.8),
      height: clamp(height / 24, 0.9, 3.8),
      depth: clamp(channels / 10, 0.28, 1.8)
    };
  }

  if (shape.length === 2) {
    const [rows, cols] = shape;
    return {
      width: clamp(cols / 24, 0.8, 3.3),
      height: clamp(rows / 24, 0.8, 3.3),
      depth: 0.35
    };
  }

  const units = Math.max(1, shape.length === 1 ? shape[0] : layer.type === 'flatten' ? 96 : 24);
  if (layer.type === 'flatten') {
    return {
      width: clamp(Math.sqrt(units) / 3.2, 1.4, 4.2),
      height: clamp(Math.ceil(units / 24) * 0.22, 0.55, 1.35),
      depth: 0.36
    };
  }

  if (layer.type === 'dense' || layer.type === 'output') {
    const cols = Math.ceil(Math.sqrt(units));
    const rows = Math.ceil(units / cols);
    return {
      width: clamp(cols / 5, 0.9, 3),
      height: clamp(rows / 5, 0.85, 2.8),
      depth: 0.42
    };
  }

  return {
    width: clamp(Math.sqrt(units) / 4.5, 0.75, 2.6),
    height: clamp(Math.sqrt(units) / 5.5, 0.75, 2.4),
    depth: 0.38
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
