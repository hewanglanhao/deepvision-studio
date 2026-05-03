export type ModeCVisualStatus = 'ready' | 'in-progress' | 'planned';

export type ModeCFocusArea = 'overview' | 'detail' | 'article';

export interface ModeCSampleOption {
  id: string;
  title: string;
  label: string;
  description: string;
  assetPath: string;
  predictedClass: string;
  confidence: number;
  topClasses: Array<{
    label: string;
    score: number;
  }>;
}

export interface ModeCOverviewStage {
  id: string;
  title: string;
  summary: string;
  status: ModeCVisualStatus;
}

export interface ModeCDetailTopic {
  id: string;
  title: string;
  description: string;
  priority: 'P0' | 'P1' | 'P2';
}

export interface ModeCArticleSection {
  id: string;
  eyebrow: string;
  title: string;
  body: string[];
  bullets?: string[];
}

export interface ModeCMilestone {
  id: string;
  title: string;
  note: string;
  status: ModeCVisualStatus;
}

export interface ModeCModelStatusSummary {
  title: string;
  description: string;
  status: ModeCVisualStatus;
}

export interface ModeCNetworkLayer {
  id: string;
  sourceName: string;
  title: string;
  shortTitle: string;
  type: 'input' | 'conv' | 'relu' | 'pool' | 'flatten' | 'output';
  channels: number;
  spatialSize: number;
  inputShapeLabel: string;
  outputShapeLabel: string;
  kernelLabel: string | null;
  parameterCount: number;
  description: string;
  stage: 'encoder-a' | 'encoder-b' | 'bridge' | 'classifier';
}

export interface ModeCClassScore {
  label: string;
  score: number;
}

export interface ModeCLayerActivationSummary {
  layerId: string;
  min: number;
  max: number;
  mean: number;
  positiveRatio: number;
  energy: number;
}

export interface ModeCLayerPreview {
  layerId: string;
  dataUrl: string;
}

export interface ModeCLayerChannelPreview {
  index: number;
  dataUrl: string;
  matrix: number[][];
  grayscale?: boolean;
  mean: number;
  energy: number;
}

export interface ModeCLayerDetailRuntime {
  layerId: string;
  channelPreviews: ModeCLayerChannelPreview[];
  vectorValues: number[];
  convExamples?: ModeCConvChannelExample[];
  reluExamples?: ModeCReluChannelExample[];
  poolExamples?: ModeCPoolChannelExample[];
}

export interface ModeCConvChannelExample {
  outputChannelIndex: number;
  inputChannelIndex: number;
  row: number;
  col: number;
  patch: number[][];
  kernel: number[][];
  products: number[][];
  bias: number;
  weightedSum: number;
  outputValue: number;
  patchPreviewUrl: string;
  kernelPreviewUrl: string;
  inputContributions: ModeCConvInputContribution[];
}

export interface ModeCConvInputContribution {
  inputChannelIndex: number;
  contributionScore: number;
  weightedSum: number;
  patch: number[][];
  kernel: number[][];
  products: number[][];
  patchPreviewUrl: string;
  kernelPreviewUrl: string;
}

export interface ModeCReluChannelExample {
  channelIndex: number;
  beforePreviewUrl: string;
  afterPreviewUrl: string;
  beforeNegativeRatio: number;
  afterPositiveRatio: number;
  beforeMin: number;
  afterMin: number;
  afterMax: number;
}

export interface ModeCPoolChannelExample {
  channelIndex: number;
  row: number;
  col: number;
  patch: number[][];
  maxValue: number;
  patchPreviewUrl: string;
}

export interface ModeCSamplePrediction {
  label: string;
  confidence: number;
  topClasses: ModeCClassScore[];
}
