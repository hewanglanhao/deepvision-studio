import { NetworkLayer, TensorShape } from '../../sim-models';

export const NETWORK_3D_SESSION_KEY = 'deepvision-network-3d-payload';

export interface Network3dPayload {
  title: string;
  sourceMode: string;
  createdAt: string;
  inputImageUrl: string;
  inputLabel?: string;
  layers: NetworkLayer[];
  shapeHints: Record<number, string>;
  layerShapes: Record<number, TensorShape>;
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
