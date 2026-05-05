// ---------------------------------------------------------------------------
// Mode D — Pure TypeScript forward + backward propagation engine
// Zero external dependencies. Works with small networks (MLP focus).
// ---------------------------------------------------------------------------

import type {
  ModeDBackpropPhase,
  ModeDBackpropStep,
  ModeDLayerGradient,
  ModeDParameterSnapshot,
  ModeDForwardCacheEntry,
  ModeDOptimizerState,
  ModeDTrainingConfig,
  ModeDDatasetSample,
} from '../models/mode-d.types';

// ---- tiny matrix helpers ----------------------------------------------------

function vec(len: number, fill = 0): number[] {
  return Array(len).fill(fill);
}

function mat(rows: number, cols: number, fill = 0): number[][] {
  return Array.from({ length: rows }, () => vec(cols, fill));
}

function randMat(rows: number, cols: number, scale = 0.1): number[][] {
  return Array.from({ length: rows }, () =>
    vec(cols, 0).map(() => (Math.random() - 0.5) * 2 * scale)
  );
}

function randVec(len: number, scale = 0.1): number[] {
  return vec(len, 0).map(() => (Math.random() - 0.5) * 2 * scale);
}

function dot(A: number[][], B: number[][]): number[][] {
  const m = A.length;
  const n = B[0].length;
  const inner = B.length;
  const C = mat(m, n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let s = 0;
      for (let k = 0; k < inner; k++) s += A[i][k] * B[k][j];
      C[i][j] = s;
    }
  }
  return C;
}

function transpose(A: number[][]): number[][] {
  const rows = A.length;
  const cols = A[0].length;
  const T = mat(cols, rows);
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) T[j][i] = A[i][j];
  }
  return T;
}

function addVecToRows(M: number[][], b: number[]): number[][] {
  return M.map(row => row.map((v, j) => v + b[j]));
}

function subMat(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((v, j) => v - B[i][j]));
}

function mulMatScalar(A: number[][], s: number): number[][] {
  return A.map(row => row.map(v => v * s));
}

function addMat(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((v, j) => v + B[i][j]));
}

function hadamard(A: number[][], B: number[][]): number[][] {
  return A.map((row, i) => row.map((v, j) => v * B[i][j]));
}

// ---- activation functions ---------------------------------------------------

function relu(Z: number[][]): number[][] {
  return Z.map(row => row.map(v => (v > 0 ? v : 0)));
}

function reluDerivative(Z: number[][]): number[][] {
  return Z.map(row => row.map(v => (v > 0 ? 1 : 0)));
}

function sigmoid(Z: number[][]): number[][] {
  return Z.map(row => row.map(v => 1 / (1 + Math.exp(-v))));
}

function sigmoidDerivative(A: number[][]): number[][] {
  return A.map(row => row.map(v => v * (1 - v)));
}

function tanh(Z: number[][]): number[][] {
  return Z.map(row => row.map(v => Math.tanh(v)));
}

function tanhDerivative(A: number[][]): number[][] {
  return A.map(row => row.map(v => 1 - v * v));
}

function softmax(Z: number[][]): number[][] {
  // Z is [1 x n] or [batch x n]
  return Z.map(row => {
    const max = Math.max(...row);
    const exps = row.map(v => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(v => v / sum);
  });
}

function crossEntropyGradient(predictions: number[], trueLabel: number): number[] {
  const grad = [...predictions];
  grad[trueLabel] -= 1;
  return grad;
}

function mseGradient(predictions: number[], target: number[]): number[] {
  return predictions.map((p, i) => 2 * (p - target[i]) / predictions.length);
}

// ---- statistics -------------------------------------------------------------

function stats(arr: number[]): { min: number; max: number; mean: number; std: number } {
  const n = arr.length;
  if (n === 0) return { min: 0, max: 0, mean: 0, std: 0 };
  let min = Infinity;
  let max = -Infinity;
  let sum = 0;
  for (const v of arr) {
    if (v < min) min = v;
    if (v > max) max = v;
    sum += v;
  }
  const mean = sum / n;
  let ssq = 0;
  for (const v of arr) ssq += (v - mean) * (v - mean);
  const std = Math.sqrt(ssq / n);
  return { min, max, mean, std };
}

function flattenGradients(grads: number[][]): number[] {
  return grads.reduce((flat, row) => { flat.push(...row); return flat; }, [] as number[]);
}

function l2Norm(arr: number[]): number {
  return Math.sqrt(arr.reduce((s, v) => s + v * v, 0));
}

// ---- main engine class ------------------------------------------------------

export class ModeDBackpropEngine {
  // Optimizer state carried across steps
  private optState: ModeDOptimizerState = { t: 0 };

  // ---- forward pass ---------------------------------------------------------

  /**
   * Execute a complete forward pass through all layers.
   * Returns per-layer activation cache suitable for backward pass.
   */
  forwardPass(
    layers: { id: number; type: string; params: Record<string, any> }[],
    input: number[],
  ): { output: number[]; cache: ModeDForwardCacheEntry[] } {
    const cache: ModeDForwardCacheEntry[] = [];
    let current: number[][] = [input]; // batch of 1

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const prev = current;

      switch (layer.type) {
        case 'dense':
        case 'output': {
          const W: number[][] = layer.params['weights'] ?? randMat(layer.params['units'], prev[0].length, 0.1);
          const b: number[] = layer.params['bias'] ?? randVec(layer.params['units'], 0.0);
          const Z = addVecToRows(dot(prev, transpose(W)), b);
          const activation = layer.params['activation'] ?? 'relu';
          let A: number[][];
          if (layer.type === 'output' && activation === 'softmax') {
            A = softmax(Z);
          } else if (activation === 'relu') {
            A = relu(Z);
          } else if (activation === 'sigmoid') {
            A = sigmoid(Z);
          } else if (activation === 'tanh') {
            A = tanh(Z);
          } else {
            A = Z; // linear / none
          }
          current = A;
          cache.push({
            layerId: layer.id,
            layerIndex: i,
            input: prev,
            output: A,
            preActivation: Z,
          });
          break;
        }

        case 'activation': {
          const actType = layer.params['activationType'] ?? 'relu';
          let A: number[][];
          if (actType === 'relu') {
            A = relu(prev);
          } else if (actType === 'sigmoid') {
            A = sigmoid(prev);
          } else if (actType === 'tanh') {
            A = tanh(prev);
          } else {
            A = prev;
          }
          current = A;
          cache.push({
            layerId: layer.id,
            layerIndex: i,
            input: prev,
            output: A,
            preActivation: prev, // activation layer has no pre-activation separate from input
          });
          break;
        }

        case 'input':
        default: {
          // pass-through
          cache.push({
            layerId: layer.id,
            layerIndex: i,
            input: prev,
            output: prev,
          });
          break;
        }
      }
    }

    return { output: current[0], cache };
  }

  // ---- loss computation -----------------------------------------------------

  computeLoss(
    predictions: number[],
    labels: number,
    lossFunction: string,
  ): { loss: number; outputGradient: number[] } {
    const numClasses = predictions.length;

    if (lossFunction === 'crossEntropy' && numClasses > 1) {
      const eps = 1e-15;
      const p = Math.max(predictions[labels], eps);
      const loss = -Math.log(p);
      const grad = crossEntropyGradient(predictions, labels);
      return { loss, outputGradient: grad };
    }

    if (lossFunction === 'binaryCrossEntropy' && numClasses === 2) {
      const eps = 1e-15;
      const p = Math.max(predictions[labels], eps);
      const q = Math.max(1 - predictions[1 - labels], eps);
      const loss = -(Math.log(p) + Math.log(q)) / 2;
      const grad = crossEntropyGradient(predictions, labels);
      return { loss, outputGradient: grad };
    }

    if (lossFunction === 'mse') {
      const target = vec(numClasses, 0);
      target[labels] = 1;
      const loss = predictions.reduce((s, p, i) => s + (p - target[i]) ** 2, 0) / numClasses;
      const grad = mseGradient(predictions, target);
      return { loss, outputGradient: grad };
    }

    // fallback: crossEntropy
    return this.computeLoss(predictions, labels, 'crossEntropy');
  }

  // ---- backward pass --------------------------------------------------------

  backwardPass(
    layers: { id: number; type: string; params: Record<string, any> }[],
    forwardCache: ModeDForwardCacheEntry[],
    outputGradient: number[],
  ): ModeDLayerGradient[] {
    const layerGradients: ModeDLayerGradient[] = [];
    let dZ = [outputGradient]; // dL/dA for the last layer, shape [1 x n]

    // Iterate layers in reverse
    for (let i = layers.length - 1; i >= 0; i--) {
      const layer = layers[i];
      const cache = forwardCache[i];
      if (!cache) {
        layerGradients.unshift({
          layerId: layer.id,
          layerType: layer.type,
          gradientNorm: 0,
          gradientStats: { min: 0, max: 0, mean: 0, std: 0 },
        });
        continue;
      }

      switch (layer.type) {
        case 'output':
        case 'dense': {
          const W: number[][] = layer.params['weights'] ?? randMat(layer.params['units'], cache.input[0].length, 0.1);
          const activation = layer.params['activation'] ?? 'relu';

          // If activation was applied, differentiate through it first
          let dZ_local = dZ; // gradient w.r.t pre-activation Z
          if (layer.type === 'output' && activation === 'softmax') {
            // softmax + cross-entropy: dZ is already the output gradient
            // (dZ = softmax - y), no further activation derivative needed
          } else if (activation === 'relu') {
            dZ_local = hadamard(dZ, reluDerivative(cache.preActivation!));
          } else if (activation === 'sigmoid') {
            dZ_local = hadamard(dZ, sigmoidDerivative(cache.output));
          } else if (activation === 'tanh') {
            dZ_local = hadamard(dZ, tanhDerivative(cache.output));
          }
          // linear → no change

          // dW = dZ^T · A_prev  →  W is [out x in], dW should be [out x in]
          // dZ_local is [1 x out], cache.input is [1 x in]
          const dW = dot(transpose(dZ_local), cache.input); // [out x in]
          const db = dZ_local[0];
          const dA_prev = dot(dZ_local, W); // [1 x in]

          const gradFlat = flattenGradients(dW);
          const gStats = stats(gradFlat);
          const gNorm = l2Norm(gradFlat);

          layerGradients.unshift({
            layerId: layer.id,
            layerType: layer.type,
            weightGradients: dW,
            biasGradients: db,
            inputGradient: dA_prev,
            gradientNorm: gNorm,
            gradientStats: gStats,
          });

          dZ = dA_prev;
          break;
        }

        case 'activation': {
          const actType = layer.params['activationType'] ?? 'relu';
          let dA_prev: number[][];
          if (actType === 'relu') {
            dA_prev = hadamard(dZ, reluDerivative(cache.input));
          } else if (actType === 'sigmoid') {
            dA_prev = hadamard(dZ, sigmoidDerivative(cache.output));
          } else if (actType === 'tanh') {
            dA_prev = hadamard(dZ, tanhDerivative(cache.output));
          } else {
            dA_prev = dZ;
          }

          const flat = flattenGradients(dA_prev);
          const actStats = stats(flat);
          layerGradients.unshift({
            layerId: layer.id,
            layerType: layer.type,
            inputGradient: dA_prev,
            gradientNorm: l2Norm(flat),
            gradientStats: actStats,
          });

          dZ = dA_prev;
          break;
        }

        default: {
          // input / pass-through
          layerGradients.unshift({
            layerId: layer.id,
            layerType: layer.type,
            inputGradient: dZ,
            gradientNorm: 0,
            gradientStats: { min: 0, max: 0, mean: 0, std: 0 },
          });
          break;
        }
      }
    }

    return layerGradients;
  }

  // ---- optimizer step -------------------------------------------------------

  applyGradients(
    layers: { id: number; type: string; params: Record<string, any> }[],
    layerGradients: ModeDLayerGradient[],
    config: ModeDTrainingConfig,
    forwardCache: ModeDForwardCacheEntry[],
  ): ModeDParameterSnapshot[] {
    const snapshots: ModeDParameterSnapshot[] = [];
    const t = ++this.optState.t;

    // Lazy-init optimizer state
    if (config.optimizer === 'adam') {
      if (!this.optState.m) this.optState.m = {};
      if (!this.optState.v) this.optState.v = {};
    }
    if (config.optimizer === 'momentum') {
      if (!this.optState.velocity) this.optState.velocity = {};
    }

    for (let i = 0; i < layers.length; i++) {
      const layer = layers[i];
      const grad = layerGradients.find(g => g.layerId === layer.id);
      if (!grad || !grad.weightGradients) {
        // For layers without trainable params (activation, input), just snapshot current state
        snapshots.push({
          layerId: layer.id,
        });
        continue;
      }

      const W = (layer.params['weights'] as number[][]) ?? randMat(layer.params['units'], forwardCache[i]?.input[0]?.length ?? 1, 0.1);
      const b = (layer.params['bias'] as number[]) ?? randVec(layer.params['units'], 0.0);

      const wBefore = W.map(row => [...row]);
      const bBefore = [...b];

      const dW = grad.weightGradients!;
      const db = grad.biasGradients!;
      const lr = config.learningRate;

      if (config.optimizer === 'sgd') {
        for (let r = 0; r < W.length; r++) {
          for (let c = 0; c < W[r].length; c++) {
            W[r][c] -= lr * dW[r][c];
          }
          b[r] -= lr * db[r];
        }
      } else if (config.optimizer === 'momentum') {
        const vel = this.optState.velocity!;
        if (!vel[layer.id]) {
          vel[layer.id] = { w: mat(W.length, W[0].length, 0), b: vec(b.length, 0) };
        }
        const beta = 0.9;
        for (let r = 0; r < W.length; r++) {
          for (let c = 0; c < W[r].length; c++) {
            vel[layer.id].w[r][c] = beta * vel[layer.id].w[r][c] + lr * dW[r][c];
            W[r][c] -= vel[layer.id].w[r][c];
          }
          vel[layer.id].b[r] = beta * vel[layer.id].b[r] + lr * db[r];
          b[r] -= vel[layer.id].b[r];
        }
      } else {
        // adam
        const m = this.optState.m!;
        const v = this.optState.v!;
        if (!m[layer.id]) {
          m[layer.id] = { w: mat(W.length, W[0].length, 0), b: vec(b.length, 0) };
          v[layer.id] = { w: mat(W.length, W[0].length, 0), b: vec(b.length, 0) };
        }
        const beta1 = 0.9;
        const beta2 = 0.999;
        const eps = 1e-8;
        for (let r = 0; r < W.length; r++) {
          for (let c = 0; c < W[r].length; c++) {
            m[layer.id].w[r][c] = beta1 * m[layer.id].w[r][c] + (1 - beta1) * dW[r][c];
            v[layer.id].w[r][c] = beta2 * v[layer.id].w[r][c] + (1 - beta2) * dW[r][c] * dW[r][c];
            const mHat = m[layer.id].w[r][c] / (1 - Math.pow(beta1, t));
            const vHat = v[layer.id].w[r][c] / (1 - Math.pow(beta2, t));
            W[r][c] -= lr * mHat / (Math.sqrt(vHat) + eps);
          }
          m[layer.id].b[r] = beta1 * m[layer.id].b[r] + (1 - beta1) * db[r];
          v[layer.id].b[r] = beta2 * v[layer.id].b[r] + (1 - beta2) * db[r] * db[r];
          const mHatB = m[layer.id].b[r] / (1 - Math.pow(beta1, t));
          const vHatB = v[layer.id].b[r] / (1 - Math.pow(beta2, t));
          b[r] -= lr * mHatB / (Math.sqrt(vHatB) + eps);
        }
      }

      // write back
      layer.params['weights'] = W;
      layer.params['bias'] = b;

      const wAfter = W.map(row => [...row]);
      const bAfter = [...b];
      const wChange = subMat(wAfter, wBefore).map(row => row.map(v => Math.abs(v)));
      const bChange = bAfter.map((v, j) => Math.abs(v - bBefore[j]));

      snapshots.push({
        layerId: layer.id,
        weightsBefore: wBefore,
        weightsAfter: wAfter,
        biasBefore: bBefore,
        biasAfter: bAfter,
        weightChange: wChange,
        biasChange: bChange,
      });
    }

    return snapshots;
  }

  // ---- one full training step -----------------------------------------------

  trainingStep(
    layers: { id: number; type: string; params: Record<string, any> }[],
    input: number[],
    label: number,
    config: ModeDTrainingConfig,
    iteration: number,
    phaseCallback?: (phase: ModeDBackpropPhase, layerIndex: number) => void,
  ): ModeDBackpropStep {
    const totalLayers = layers.length;

    // 1. Forward pass
    phaseCallback?.('forward', 0);
    const { output, cache } = this.forwardPass(layers, input);
    const predictedClass = output.indexOf(Math.max(...output));

    // 2. Loss
    phaseCallback?.('loss', 0);
    const { loss, outputGradient } = this.computeLoss(output, label, config.lossFunction);

    // 3. Backward pass
    phaseCallback?.('backward', 0);
    const layerGradients = this.backwardPass(layers, cache, outputGradient);

    // 4. Update
    phaseCallback?.('update', 0);
    const parameterSnapshots = this.applyGradients(layers, layerGradients, config, cache);

    return {
      iteration,
      phase: 'update',
      layerIndex: totalLayers - 1,
      totalLayers,
      forwardCache: cache,
      loss,
      predictedClass,
      trueClass: label,
      predictions: output,
      layerGradients,
      parameterSnapshots,
      optimizerState: this.optState,
    };
  }

  // ---- decision boundary ----------------------------------------------------

  computeDecisionBoundary(
    layers: { id: number; type: string; params: Record<string, any> }[],
    resolution: number,
    xRange: [number, number],
    yRange: [number, number],
  ): { resolution: number; xMin: number; xMax: number; yMin: number; yMax: number; grid: number[][] } {
    const grid: number[][] = [];
    const [xMin, xMax] = xRange;
    const [yMin, yMax] = yRange;
    const dx = (xMax - xMin) / (resolution - 1);
    const dy = (yMax - yMin) / (resolution - 1);

    for (let yi = 0; yi < resolution; yi++) {
      const row: number[] = [];
      const y = yMin + yi * dy;
      for (let xi = 0; xi < resolution; xi++) {
        const x = xMin + xi * dx;
        const { output } = this.forwardPass(layers, [x, y]);
        const cls = output.indexOf(Math.max(...output));
        row.push(cls);
      }
      grid.push(row);
    }

    return { resolution, xMin, xMax, yMin, yMax, grid };
  }

  // ---- reset optimizer state ------------------------------------------------

  reset(): void {
    this.optState = { t: 0 };
  }

  // ---- dataset generators ---------------------------------------------------

  static generateXorData(n: number = 200, noise: number = 0.05): ModeDDatasetSample[] {
    const samples: ModeDDatasetSample[] = [];
    const half = Math.floor(n / 4);
    const centers: [number, number, number][] = [
      [0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1],
    ];
    for (const [cx, cy, label] of centers) {
      for (let i = 0; i < half; i++) {
        const x = cx + (Math.random() - 0.5) * 2 * noise;
        const y = cy + (Math.random() - 0.5) * 2 * noise;
        samples.push({ input: [x, y], label });
      }
    }
    return samples;
  }

  static generateCircleData(n: number = 300, noise: number = 0.1): ModeDDatasetSample[] {
    const samples: ModeDDatasetSample[] = [];
    const half = Math.floor(n / 2);
    for (let i = 0; i < half; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const r = 0.3 + (Math.random() - 0.5) * 2 * noise;
      samples.push({ input: [0.5 + r * Math.cos(angle), 0.5 + r * Math.sin(angle)], label: 0 });
    }
    for (let i = 0; i < half; i++) {
      const angle = Math.random() * 2 * Math.PI;
      const r = 0.65 + (Math.random() - 0.5) * 2 * noise;
      samples.push({ input: [0.5 + r * Math.cos(angle), 0.5 + r * Math.sin(angle)], label: 1 });
    }
    return samples;
  }

  static generateSpiralData(n: number = 400, classes: number = 3, noise: number = 0.2): ModeDDatasetSample[] {
    const samples: ModeDDatasetSample[] = [];
    const perClass = Math.floor(n / classes);
    for (let c = 0; c < classes; c++) {
      for (let i = 0; i < perClass; i++) {
        const t = (i / perClass) * 2.5 * Math.PI;
        const r = t / (2.5 * Math.PI);
        const angle = t + (c * 2 * Math.PI) / classes;
        const x = 0.5 + r * Math.cos(angle) * 0.45 + (Math.random() - 0.5) * noise;
        const y = 0.5 + r * Math.sin(angle) * 0.45 + (Math.random() - 0.5) * noise;
        samples.push({ input: [x, y], label: c });
      }
    }
    return samples;
  }

  static generateBlobData(n: number = 400, centers: [number, number][] = [[0.3, 0.3], [0.7, 0.7], [0.3, 0.7]], noise: number = 0.08): ModeDDatasetSample[] {
    const samples: ModeDDatasetSample[] = [];
    const perBlob = Math.floor(n / centers.length);
    for (let c = 0; c < centers.length; c++) {
      const [cx, cy] = centers[c];
      for (let i = 0; i < perBlob; i++) {
        const x = cx + (Math.random() - 0.5) * 2 * noise;
        const y = cy + (Math.random() - 0.5) * 2 * noise;
        samples.push({ input: [x, y], label: c });
      }
    }
    return samples;
  }
}
