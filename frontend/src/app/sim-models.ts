export type LayerType =
  | 'input'
  | 'conv2d'
  | 'pool2d'
  | 'residual'
  | 'flatten'
  | 'dense'
  | 'activation'
  | 'dropout'
  | 'output';

export type ActivationType = 'none' | 'relu' | 'tanh' | 'gelu' | 'sigmoid' | 'softmax';
export type ColorMode = 'grayscale' | 'rgb';
export type InputKind = 'image' | 'table';
export type PoolMode = 'max' | 'avg';
export type ResizeMode = 'none' | 'fit';

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
export type AppMode = 'forward' | 'training';

export type TensorShape = [] | [number] | [number, number] | [number, number, number];
export type TensorKind = 'scalar' | 'vector' | 'matrix' | 'tensor3d';

export interface Connection {
  from: number;
  to: number;
}

export interface LayerDisplay {
  accent?: string;
  shortLabel?: string;
}

export interface InputPreprocessConfig {
  resizeMode: ResizeMode;
  targetWidth?: number;
  targetHeight?: number;
  colorMode: 'original' | ColorMode;
  normalize: 'none' | 'zero-one';
  invert: boolean;
}

export interface InputLayerParams {
  inputKind: InputKind;
  width: number;
  height: number;
  channels: number;
  featureCount: number;
  colorMode: ColorMode;
  preprocessing: InputPreprocessConfig;
}

export interface ConvKernelSpec {
  weights: number[][][];
  bias?: number;
  label?: string;
}

export interface Conv2DLayerParams {
  outChannels: number;
  kernelSize: number;
  stride: number;
  padding: number;
  dilation: number;
  kernelMatrix?: number[][];
  kernels?: ConvKernelSpec[];
  bias?: number[];
  activation: ActivationType;
}

export interface Pool2DLayerParams {
  mode: PoolMode;
  kernelSize: number;
  stride: number;
  padding: number;
}

export interface ResidualBlockLayerParams {
  outChannels: number;
  kernelSize: number;
  stride: number;
  padding: number;
  activation: Exclude<ActivationType, 'softmax'>;
  useProjection: boolean;
}

export interface FlattenLayerParams {}

export interface DenseLayerParams {
  units: number;
  weights?: number[][];
  bias?: number[];
  activation: ActivationType;
}

export interface ActivationLayerParams {
  activationType: ActivationType;
}

export interface DropoutLayerParams {
  rate: number;
  training?: boolean;
}

export interface OutputLayerParams {
  units: number;
  weights?: number[][];
  bias?: number[];
  activation: ActivationType;
  labels?: string[];
}

export interface BaseLayer<TType extends LayerType, TParams> {
  id: number;
  type: TType;
  name: string;
  inputs: number[];
  params: TParams;
  display?: LayerDisplay;
  enabled?: boolean;
}

export type InputLayer = BaseLayer<'input', InputLayerParams>;
export type Conv2DLayer = BaseLayer<'conv2d', Conv2DLayerParams>;
export type Pool2DLayer = BaseLayer<'pool2d', Pool2DLayerParams>;
export type ResidualBlockLayer = BaseLayer<'residual', ResidualBlockLayerParams>;
export type FlattenLayer = BaseLayer<'flatten', FlattenLayerParams>;
export type DenseLayer = BaseLayer<'dense', DenseLayerParams>;
export type ActivationLayer = BaseLayer<'activation', ActivationLayerParams>;
export type DropoutLayer = BaseLayer<'dropout', DropoutLayerParams>;
export type OutputLayer = BaseLayer<'output', OutputLayerParams>;

export type NetworkLayer =
  | InputLayer
  | Conv2DLayer
  | Pool2DLayer
  | ResidualBlockLayer
  | FlattenLayer
  | DenseLayer
  | ActivationLayer
  | DropoutLayer
  | OutputLayer;

export type LayerDraft = Omit<NetworkLayer, 'id'>;

export interface DataSample {
  id: number;
  label: number;
  pixels: number[];
  width: number;
  height: number;
  channels: number;
  colorMode: ColorMode;
  previewPixels: number[];
}

export interface ForwardTensor {
  kind: TensorKind;
  shape: TensorShape;
  values: number[];
  colorMode?: ColorMode;
  labels?: string[];
}

export interface TensorStats {
  min: number;
  max: number;
  mean: number;
  nonZeroRatio: number;
  topK: Array<{ index: number; value: number; label?: string }>;
}

export interface ImageChannelPreview {
  channel: number;
  width: number;
  height: number;
  values: number[];
}

export interface LayerVisualization {
  mode: 'image' | 'vector' | 'none';
  width?: number;
  height?: number;
  channels?: number;
  values: number[];
  channelPreviews?: ImageChannelPreview[];
}

export interface LayerValidationIssue {
  layerId: number;
  layerName: string;
  severity: 'error' | 'warning';
  message: string;
  field?: string;
}

export interface ForwardLayerResult {
  layerId: number;
  layerName: string;
  layerType: LayerType;
  inputShapes: TensorShape[];
  outputShape: TensorShape;
  inputShapeLabel: string;
  outputShapeLabel: string;
  shapeLabel: string;
  transitionNote: string;
  paramsSummary: string[];
  warnings: string[];
  tensor: ForwardTensor;
  visualization: LayerVisualization;
  stats: TensorStats;
}

export interface ForwardPassResult {
  executionOrder: number[];
  layerResults: ForwardLayerResult[];
  layerShapeMap: Record<number, string>;
  finalTensor: ForwardTensor | null;
  finalTopK: Array<{ index: number; value: number; label?: string }>;
  validationIssues: LayerValidationIssue[];
  shapePath: string[];
  errors: string[];
  warnings: string[];
  resolvedLayers: NetworkLayer[];
}

export interface PreparedInputAsset {
  tensor: ForwardTensor;
  displayTensor: ForwardTensor;
  notes: string[];
}

export interface ForwardInputAsset {
  id: string;
  source: 'dataset' | 'upload';
  name: string;
  previewUrl?: string;
  originalWidth: number;
  originalHeight: number;
  originalChannels: number;
  originalColorMode: ColorMode;
  originalTensor: ForwardTensor;
  prepared: PreparedInputAsset;
  label?: string;
}

export interface TrainingState {
  status: 'idle' | 'running' | 'paused' | 'stopped' | 'completed';
  currentEpoch: number;
  currentLr: number;
  latestLoss: number;
  latestValLoss: number;
  latestAccuracy: number;
  latestValAccuracy: number;
  latestGradientNorm: number;
  latestWeightMean: number;
  latestWeightStd: number;
  elapsedSeconds: number;
  etaSeconds: number;
  currentBatch?: number;
  totalBatches?: number;
}

export interface ForwardModeState {
  message: string;
  status: 'idle' | 'ready' | 'error';
  errors: string[];
}

export interface MetricPoint {
  step: number;
  loss: number;
  valLoss: number;
  accuracy: number;
  valAccuracy: number;
  lr: number;
  gradientNorm: number;
  weightMean: number;
  weightStd: number;
  elapsedSeconds: number;
  etaSeconds: number;
}

export interface TrainingConfig {
  batchSize: number;
  totalEpochs: number;
  learningRate: number;
  optimizer: OptimizerType;
  scheduler: SchedulerType;
  lrDecay: number;
}

export interface TrainingRuntimeState extends TrainingState {
  message: string;
}

export interface TrainingDataInfo {
  source: 'builtin' | 'uploaded';
  hasLabels: boolean;
  sampleCount: number;
}

export type TrainingDatasetSource = 'builtin' | 'upload';
export type TrainingDatasetKind = 'image' | 'table' | 'points';
export type DatasetUploadStatus = 'idle' | 'pending' | 'ready' | 'error';

export interface TrainingDatasetOption {
  id: string;
  name: string;
  source: TrainingDatasetSource;
  kind: TrainingDatasetKind;
  description: string;
  sampleCount: number;
  classCount: number;
  inputShape: string;
  recommendedSplit: string;
  labels: string[];
}

export interface LabelDistributionItem {
  label: string;
  count: number;
  color: string;
}

export interface TablePreview {
  headers: string[];
  rows: string[][];
}

export interface ImagePreviewItem {
  name: string;
  label: string;
  url: string;
}

export interface PointPreviewItem {
  x: number;
  y: number;
  label: string;
  color: string;
}

export interface TrainingDatasetDetail extends TrainingDatasetOption {
  hasLabels: boolean;
  trainRatio: number;
  valRatio: number;
  testRatio: number;
  labelDistribution: LabelDistributionItem[];
  tablePreview?: TablePreview;
  imagePreview?: ImagePreviewItem[];
  pointPreview?: PointPreviewItem[];
  warnings: string[];
}

export interface DatasetImportDraft {
  status: DatasetUploadStatus;
  message: string;
  files: File[];
  detectedKind: TrainingDatasetKind | null;
  detail: TrainingDatasetDetail | null;
  csvHeaders: string[];
  selectedLabelColumn: string;
  selectedClassCount: number | null;
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
  layers: LayerDraft[];
}
