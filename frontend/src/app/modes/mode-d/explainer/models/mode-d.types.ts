// ---------------------------------------------------------------------------
// Mode D — Backpropagation Visualization
// Domain types for the backprop explainer feature.
// ---------------------------------------------------------------------------

import type { NetworkLayer } from '@shared/simulation/sim-models';

// ---- UI meta ----------------------------------------------------------------

export type ModeDVisualStatus = 'idle' | 'ready' | 'running' | 'paused';
export type ModeDFocusArea = 'overview' | 'detail' | 'controls' | 'article';
export type ModeDBackpropPhase = 'forward' | 'loss' | 'backward' | 'update';
export type ModeDLossFunction = 'crossEntropy' | 'mse' | 'binaryCrossEntropy';
export type ModeDOptimizer = 'sgd' | 'momentum' | 'adam';

// ---- Dataset ----------------------------------------------------------------

export interface ModeDDatasetSample {
  input: number[];
  label: number;
}

export interface ModeDDatasetPreset {
  id: string;
  name: string;
  description: string;
  samples: ModeDDatasetSample[];
  inputDim: number;
  outputDim: number;
  classLabels: string[];
}

// ---- Network preset ---------------------------------------------------------

export interface ModeDNetworkPreset {
  id: string;
  name: string;
  description: string;
  layers: NetworkLayer[];
  connections: { from: number; to: number }[];
  datasetId: string;
}

// ---- Training config --------------------------------------------------------

export interface ModeDTrainingConfig {
  learningRate: number;
  optimizer: ModeDOptimizer;
  lossFunction: ModeDLossFunction;
  maxIterations: number;
}

// ---- Forward-pass cache (per layer) -----------------------------------------

export interface ModeDForwardCacheEntry {
  layerId: number;
  layerIndex: number;
  input: number[][];       // pre-activation (or input tensor for non-dense layers)
  output: number[][];      // post-activation
  preActivation?: number[][]; // Z = W·X + b  (before activation)
}

// ---- Gradient tracking ------------------------------------------------------

export interface ModeDLayerGradient {
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

export interface ModeDParameterSnapshot {
  layerId: number;
  weightsBefore?: number[][];
  weightsAfter?: number[][];
  biasBefore?: number[];
  biasAfter?: number[];
  weightChange?: number[][];  // element-wise absolute difference
  biasChange?: number[];       // element-wise absolute difference
}

// ---- Optimizer state --------------------------------------------------------

export interface ModeDOptimizerState {
  // Adam moment estimates, keyed by layerId
  m?: Record<number, { w: number[][]; b: number[] }>;
  v?: Record<number, { w: number[][]; b: number[] }>;
  // Momentum velocity buffers
  velocity?: Record<number, { w: number[][]; b: number[] }>;
  t: number; // step counter
}

// ---- Per-step result --------------------------------------------------------

export interface ModeDBackpropStep {
  iteration: number;
  phase: ModeDBackpropPhase;
  layerIndex: number;          // which layer is currently being processed
  totalLayers: number;
  forwardCache?: ModeDForwardCacheEntry[];
  loss?: number;
  predictedClass?: number;
  trueClass?: number;
  predictions?: number[];      // softmax output
  layerGradients: ModeDLayerGradient[];
  parameterSnapshots: ModeDParameterSnapshot[];
  optimizerState?: ModeDOptimizerState;
}

// ---- Layer activation summary (for visualization) ---------------------------

export interface ModeDLayerActivationSummary {
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

export interface ModeDDecisionBoundary {
  // grid of predictions over the 2D input space
  resolution: number;          // e.g. 50 → 50×50 grid
  xMin: number;
  xMax: number;
  yMin: number;
  yMax: number;
  grid: number[][];            // grid[y][x] = predicted class index
}

// ---- Weight matrix visualization data ---------------------------------------

export interface ModeDMatrixViewData {
  values: number[][];
  rows: number;
  cols: number;
  min: number;
  max: number;
  colorMode: 'diverging' | 'sequential';  // diverging for gradients, sequential for weights
}

// ---- Gradient flow edge -----------------------------------------------------

export interface ModeDGradientFlowEdge {
  fromLayerId: string;
  toLayerId: string;
  magnitude: number;           // normalized gradient norm
  status: 'vanishing' | 'stable' | 'exploding';
}

// ---- MSE-specific loss breakdown --------------------------------------------

export interface ModeDLossBreakdown {
  total: number;
  perSample?: number[];
  gradientAtOutput?: number[];  // dL/dy for the last layer
}

// ---- Home / card status -----------------------------------------------------

export interface ModeDCardStatus {
  title: string;
  label: string;
  description: string;
  route: string;
  status: string;
}
