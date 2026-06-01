// ---------------------------------------------------------------------------
// Mode E — Backpropagation Visualization
// Domain types for the backprop explainer feature.
// ---------------------------------------------------------------------------

import type { NetworkLayer } from '@shared/simulation/sim-models';

// ---- UI meta ----------------------------------------------------------------

export type ModeEVisualStatus = 'idle' | 'ready' | 'running' | 'paused';
export type ModeEFocusArea = 'overview' | 'detail' | 'controls' | 'article';
export type ModeEBackpropPhase = 'forward' | 'loss' | 'backward' | 'update';
export type ModeELossFunction = 'crossEntropy' | 'mse' | 'binaryCrossEntropy';
export type ModeEOptimizer = 'sgd' | 'momentum' | 'adam';

// ---- Dataset ----------------------------------------------------------------

export interface ModeEDatasetSample {
  input: number[];
  label: number;
}

export interface ModeEDatasetPreset {
  id: string;
  name: string;
  description: string;
  samples: ModeEDatasetSample[];
  inputDim: number;
  outputDim: number;
  classLabels: string[];
}

// ---- Network preset ---------------------------------------------------------

export interface ModeENetworkPreset {
  id: string;
  name: string;
  description: string;
  layers: NetworkLayer[];
  connections: { from: number; to: number }[];
  datasetId: string;
}

// ---- Training config --------------------------------------------------------

export interface ModeETrainingConfig {
  learningRate: number;
  optimizer: ModeEOptimizer;
  lossFunction: ModeELossFunction;
  maxIterations: number;
}

// ---- Forward-pass cache (per layer) -----------------------------------------

export interface ModeEForwardCacheEntry {
  layerId: number;
  layerIndex: number;
  input: number[][];       // pre-activation (or input tensor for non-dense layers)
  output: number[][];      // post-activation
  preActivation?: number[][]; // Z = W.X + b  (before activation)
}

// ---- Gradient tracking ------------------------------------------------------

export interface ModeELayerGradient {
  layerId: number;
  layerType: string;
  weightGradients?: number[][];   // dW  matrix
  biasGradients?: number[];       // db  vector
  inputGradient?: number[][];     // dA_prev — propagated upstream
  gradientNorm: number;
  gradientStats: {
    min: number;
    max: number;
    mean: number;
    std: number;
  };
}

// ---- Parameter snapshot (before / after an optimizer step) ------------------

export interface ModeEParameterSnapshot {
  layerId: number;
  weightsBefore?: number[][];
  weightsAfter?: number[][];
  biasBefore?: number[];
  biasAfter?: number[];
  weightChange?: number[][];  // element-wise absolute difference
  biasChange?: number[];       // element-wise absolute difference
}

// ---- Optimizer state --------------------------------------------------------

export interface ModeEOptimizerState {
  // Adam moment estimates, keyed by layerId
  m?: Record<number, { w: number[][]; b: number[] }>;
  v?: Record<number, { w: number[][]; b: number[] }>;
  // Momentum velocity buffers
  velocity?: Record<number, { w: number[][]; b: number[] }>;
  t: number; // step counter
}

// ---- Per-step result --------------------------------------------------------

export interface ModeEBackpropStep {
  iteration: number;
  phase: ModeEBackpropPhase;
  layerIndex: number;          // which layer is currently being processed
  totalLayers: number;
  forwardCache?: ModeEForwardCacheEntry[];
  loss?: number;
  predictedClass?: number;
  trueClass?: number;
  predictions?: number[];      // softmax output
  layerGradients: ModeELayerGradient[];
  parameterSnapshots: ModeEParameterSnapshot[];
  optimizerState?: ModeEOptimizerState;
}

// ---- Layer activation summary (for visualization) ---------------------------

export interface ModeELayerActivationSummary {
  layerId: number;
  layerName: string;
  layerType: string;
  neuronCount: number;
  activations: number[];       // flattened activation values for the current sample
  min: number;
  max: number;
  mean: number;
}

// ---- Decision boundary data -------------------------------------------------

export interface ModeEDecisionBoundary {
  // grid of predictions over the 2D input space
  resolution: number;          // e.g. 50 -> 50x50 grid
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  grid: number[][];            // grid[y][x] = predicted class index
}

// ---- Weight matrix visualization data ---------------------------------------

export interface ModeEMatrixViewData {
  values: number[][];
  rows: number;
  cols: number;
  min: number;
  max: number;
  colorMode: 'diverging' | 'sequential';  // diverging for gradients, sequential for weights
}

// ---- Gradient flow edge -----------------------------------------------------

export interface ModeEGradientFlowEdge {
  fromLayerId: string;
  toLayerId: string;
  magnitude: number;           // normalized gradient norm
  status: 'vanishing' | 'stable' | 'exploding';
}

// ---- MSE-specific loss breakdown --------------------------------------------

export interface ModeELossBreakdown {
  total: number;
  perSample?: number[];
  gradientAtOutput?: number[];  // dL/dy for the last layer
}

// ---- Home / card status -----------------------------------------------------

export interface ModeECardStatus {
  title: string;
  label: string;
  description: string;
  route: string;
  status: string;
}
