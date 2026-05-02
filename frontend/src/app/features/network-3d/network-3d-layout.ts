import { NetworkLayer, TensorShape } from '../../sim-models';
import { Network3dLayerView } from './network-3d.models';

const LAYER_COLORS: Record<string, string> = {
  input: '#6366f1',
  conv2d: '#0ea5e9',
  pool2d: '#10b981',
  flatten: '#f59e0b',
  dense: '#8b5cf6',
  activation: '#ec4899',
  dropout: '#94a3b8',
  output: '#ef4444'
};

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

  const units = shape.length === 1 ? shape[0] : layer.type === 'flatten' ? 96 : 24;
  return {
    width: clamp(Math.sqrt(Math.max(1, units)) / 5, 0.55, 2.4),
    height: clamp(units / 80, 0.8, 3.5),
    depth: 0.5
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}
