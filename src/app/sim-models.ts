export type LayerType = 'input' | 'dense' | 'conv' | 'pool' | 'dropout' | 'output';

export type OptimizerType =
  | 'Adam'
  | 'AdamW'
  | 'RMSProp'
  | 'SGD'
  | 'Momentum'
  | 'Nesterov'
  | 'Adagrad'
  | 'Adadelta';

export type SchedulerType = 'none' | 'step' | 'cosine';

export interface NetworkLayer {
  id: number;
  type: LayerType;
  name: string;
  units: number;
  kernelSize: number;
  activation: string;
  dropoutRate: number;
}

export interface Connection {
  from: number;
  to: number;
}

export interface DataSample {
  id: number;
  label: number;
  pixels: number[];
}

export interface MetricPoint {
  step: number;
  loss: number;
  accuracy: number;
  valAccuracy: number;
  lr: number;
}

export interface PresetTask {
  id: string;
  name: string;
  type: 'classification' | 'regression';
  dataset: string;
  description: string;
}

export interface ExperimentResult {
  name: string;
  epochs: number;
  finalAccuracy: number;
  speedScore: number;
}

export interface ModelTemplate {
  id: string;
  name: string;
  description: string;
  layers: Array<Omit<NetworkLayer, 'id'>>;
}
