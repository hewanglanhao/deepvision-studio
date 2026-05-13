import { NetworkLayer, TensorShape, TensorStats } from '@shared/simulation/sim-models';

export const NETWORK_3D_SESSION_KEY = 'deepvision-network-3d-payload';

export interface Network3dChannelPreview {
  channel: number;
  width: number;
  height: number;
  imageUrl: string;
}

export interface Network3dLayerSnapshot {
  layerId: number;
  inputShapeLabel: string;
  outputShapeLabel: string;
  transitionNote: string;
  paramsSummary: string[];
  warnings: string[];
  stats: TensorStats;
  visualizationMode: 'image' | 'vector' | 'none';
  previewImageUrl?: string;
  channelPreviews: Network3dChannelPreview[];
  topK: Array<{ index: number; value: number; label?: string }>;
}

export interface Network3dPayload {
  title: string;
  sourceMode: string;
  createdAt: string;
  inputImageUrl: string;
  inputLabel?: string;
  datasetName?: string;
  parameterCount?: number;
  layers: NetworkLayer[];
  shapeHints: Record<number, string>;
  layerShapes: Record<number, TensorShape>;
  layerSnapshots?: Record<number, Network3dLayerSnapshot>;
  shapePath?: string[];
  finalTopK?: Array<{ index: number; value: number; label?: string }>;
  selectedLayerId: number;
}

export interface Network3dLayerView {
  layer: NetworkLayer;
  shape: TensorShape;
  shapeLabel: string;
  width: number;
  height: number;
  depth: number;
  color: string;
}
