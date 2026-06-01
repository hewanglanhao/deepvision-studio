// Mode F — RNN Learning Visualization
export type ModeFRnnCell = 'simple' | 'tanh';

export interface ModeFSequenceSample {
  id: number;
  inputs: number[][];  // [timeStep][inputDim]
  label: number;       // classification target
  labelName?: string;
}

export interface ModeFDatasetPreset {
  id: string;
  name: string;
  description: string;
  samples: ModeFSequenceSample[];
  inputDim: number;
  hiddenDim: number;
  outputDim: number;
  classLabels: string[];
  maxTimeSteps: number;
}

export interface ModeFNetworkPreset {
  id: string;
  name: string;
  description: string;
  cellType: ModeFRnnCell;
  inputDim: number;
  hiddenDim: number;
  outputDim: number;
  datasetId: string;
}

export interface ModeFTrainingConfig {
  learningRate: number;
  optimizer: 'sgd' | 'momentum' | 'adam';
  maxIterations: number;
}

export interface ModeFRnnState {
  hidden: number[];   // current hidden state
  output: number[];   // current output
}

export interface ModeFForwardResult {
  states: ModeFRnnState[];       // per-timestep state
  finalPrediction: number[];     // softmax output at last step
  predictions: number[][];       // output at each step
}

export interface ModeFRnnGradient {
  dWhy: number[][];    // hidden→output weights
  dWhh: number[][];    // hidden→hidden weights
  dWxh: number[][];    // input→hidden weights
  dbh: number[];       // hidden bias
  dby: number[];       // output bias
  gradientNorm: number;
}

export interface ModeFStepResult {
  iteration: number;
  loss: number;
  predictedClass: number;
  trueClass: number;
  forwardResult: ModeFForwardResult;
  gradient?: ModeFRnnGradient;
  hiddenDim: number;
  timeSteps: number;
  outputProbs: number[];
}

export type ModeFVisualStatus = 'idle' | 'ready' | 'running' | 'paused';
export type ModeFFocusArea = 'overview' | 'detail' | 'controls';
