// Mode E — Pure TypeScript MLP engine with incremental sub-step support
// Zero external dependencies.

import type { ModeEBackpropPhase, ModeEBackpropStep, ModeELayerGradient, ModeEParameterSnapshot,
  ModeEForwardCacheEntry, ModeEOptimizerState, ModeETrainingConfig, ModeEDatasetSample,
} from '../models/mode-e.types';

// ---- tiny matrix helpers (unchanged) ----------------------------------------
function vec(n: number, f = 0): number[] { return Array(n).fill(f); }
function mat(r: number, c: number, f = 0): number[][] { return Array.from({ length: r }, () => vec(c, f)); }
function randMat(r: number, c: number, s = 0.1): number[][] { return mat(r, c).map(row => row.map(() => (Math.random() - 0.5) * 2 * s)); }
function randVec(n: number, s = 0.1): number[] { return vec(n).map(() => (Math.random() - 0.5) * 2 * s); }
function dot(A: number[][], B: number[][]): number[][] { const m = A.length, n = B[0].length, p = B.length, C = mat(m, n); for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) for (let k = 0; k < p; k++) C[i][j] += A[i][k] * B[k][j]; return C; }
function transpose(A: number[][]): number[][] { const r = A.length, c = A[0].length, T = mat(c, r); for (let i = 0; i < r; i++) for (let j = 0; j < c; j++) T[j][i] = A[i][j]; return T; }
function addVecToRows(M: number[][], b: number[]): number[][] { return M.map(r => r.map((v, j) => v + b[j])); }
function subMat(A: number[][], B: number[][]): number[][] { return A.map((r, i) => r.map((v, j) => v - B[i][j])); }
function hadamard(A: number[][], B: number[][]): number[][] { return A.map((r, i) => r.map((v, j) => v * B[i][j])); }
function relu(Z: number[][]): number[][] { return Z.map(r => r.map(v => v > 0 ? v : 0)); }
function reluDeriv(Z: number[][]): number[][] { return Z.map(r => r.map(v => v > 0 ? 1 : 0)); }
function sigmoid(Z: number[][]): number[][] { return Z.map(r => r.map(v => 1 / (1 + Math.exp(-v)))); }
function sigmoidDeriv(A: number[][]): number[][] { return A.map(r => r.map(v => v * (1 - v))); }
function ttanh(Z: number[][]): number[][] { return Z.map(r => r.map(v => Math.tanh(v))); }
function tanhDeriv(A: number[][]): number[][] { return A.map(r => r.map(v => 1 - v * v)); }
function softmax(Z: number[][]): number[][] { return Z.map(row => { const max = Math.max(...row); const exps = row.map(v => Math.exp(v - max)); const s = exps.reduce((a, b) => a + b, 0); return exps.map(v => v / s); }); }
function ceGrad(pred: number[], label: number): number[] { const g = [...pred]; g[label] -= 1; return g; }
function mseGrad(pred: number[], tgt: number[]): number[] { return pred.map((p, i) => 2 * (p - tgt[i]) / pred.length); }
function stats(arr: number[]): { min: number; max: number; mean: number; std: number } { const n = arr.length; if (!n) return { min: 0, max: 0, mean: 0, std: 0 }; let min = Infinity, max = -Infinity, sum = 0; for (const v of arr) { if (v < min) min = v; if (v > max) max = v; sum += v; } const mean = sum / n; let ssq = 0; for (const v of arr) ssq += (v - mean) ** 2; return { min, max, mean, std: Math.sqrt(ssq / n) }; }
function l2Norm(arr: number[]): number { return Math.sqrt(arr.reduce((s, v) => s + v * v, 0)); }

// ---- activation helpers -----------------------------------------------------
function activate(Z: number[][], act: string, isOutput: boolean): number[][] {
  if (isOutput && act === 'softmax') return softmax(Z);
  if (act === 'relu') return relu(Z); if (act === 'sigmoid') return sigmoid(Z);
  if (act === 'tanh') return ttanh(Z); return Z;
}
function actDeriv(dZ: number[][], cache: ModeEForwardCacheEntry, act: string, isOutput: boolean): number[][] {
  if (isOutput && act === 'softmax') return dZ;
  if (act === 'relu') return hadamard(dZ, reluDeriv(cache.preActivation!));
  if (act === 'sigmoid') return hadamard(dZ, sigmoidDeriv(cache.output));
  if (act === 'tanh') return hadamard(dZ, tanhDeriv(cache.output));
  return dZ;
}

// ---- optimizer helpers ------------------------------------------------------
function applyOpt(W: number[][], dW: number[][], o: number[][], v: number[][], b: number[], db: number[], ob: number[], vb: number[], opt: string, lr: number, t: number): void {
  if (opt === 'sgd') { for (let i = 0; i < W.length; i++) { for (let j = 0; j < W[i].length; j++) W[i][j] -= lr * dW[i][j]; b[i] -= lr * db[i]; } return; }
  if (opt === 'momentum') { const beta = 0.9; if (o.length === 0) { for (let i = 0; i < W.length; i++) { o[i] = vec(W[i].length); ob[i] = 0; } }
    for (let i = 0; i < W.length; i++) { for (let j = 0; j < W[i].length; j++) { o[i][j] = beta * o[i][j] + lr * dW[i][j]; W[i][j] -= o[i][j]; } ob[i] = beta * ob[i] + lr * db[i]; b[i] -= ob[i]; } return; }
  // adam
  const b1 = 0.9, b2 = 0.999, eps = 1e-8; if (o.length === 0) { for (let i = 0; i < W.length; i++) { o[i] = vec(W[i].length); v[i] = vec(W[i].length); ob[i] = 0; vb[i] = 0; } }
  for (let i = 0; i < W.length; i++) { for (let j = 0; j < W[i].length; j++) { o[i][j] = b1 * o[i][j] + (1 - b1) * dW[i][j]; v[i][j] = b2 * v[i][j] + (1 - b2) * dW[i][j] * dW[i][j]; const mh = o[i][j] / (1 - b1 ** t), vh = v[i][j] / (1 - b2 ** t); W[i][j] -= lr * mh / (Math.sqrt(vh) + eps); } ob[i] = b1 * ob[i] + (1 - b1) * db[i]; vb[i] = b2 * vb[i] + (1 - b2) * db[i] * db[i]; const mhb = ob[i] / (1 - b1 ** t), vhb = vb[i] / (1 - b2 ** t); b[i] -= lr * mhb / (Math.sqrt(vhb) + eps); }
}

// ---- engine class -----------------------------------------------------------
export class ModeEBackpropEngine {
  // ---- incremental step state (shared across sub-steps) --------------------
  incCache: ModeEForwardCacheEntry[] = [];
  incGradients: ModeELayerGradient[] = [];
  incSnapshots: ModeEParameterSnapshot[] = [];
  incOutputGrad: number[] = [];
  incLoss = 0;
  incPredClass = 0;
  incTrueClass = 0;
  incPredictions: number[] = [];
  incBackwardDZ: number[][] = [];      // current dZ being propagated backward
  incOptT = 0;
  incOptM: Record<number, { w: number[][]; b: number[] }> = {};
  incOptV: Record<number, { w: number[][]; b: number[] }> = {};
  incOptVel: Record<number, { w: number[][]; b: number[] }> = {};

  private layers: { id: number; type: string; params: Record<string, any> }[] = [];
  private config: ModeETrainingConfig = { learningRate: 0.1, optimizer: 'adam', lossFunction: 'crossEntropy', maxIterations: 1000 };
  private input: number[] = [];
  private label = 0;

  // ---- full-step methods (for instantStep / decisionBoundary) --------------
  forwardPass(layers: { id: number; type: string; params: Record<string, any> }[], input: number[]): { output: number[]; cache: ModeEForwardCacheEntry[] } {
    const cache: ModeEForwardCacheEntry[] = []; let cur: number[][] = [input];
    for (let i = 0; i < layers.length; i++) {
      const l = layers[i], prev = cur;
      if (l.type === 'dense' || l.type === 'output') {
        if (!l.params['weights']) l.params['weights'] = randMat(l.params['units'], prev[0].length, 0.1);
        if (!l.params['bias']) l.params['bias'] = randVec(l.params['units'], 0);
        const W = l.params['weights'] as number[][];
        const b = l.params['bias'] as number[];
        const Z = addVecToRows(dot(prev, transpose(W)), b);
        const act = l.params['activation'] ?? 'relu'; cur = activate(Z, act, l.type === 'output');
        cache.push({ layerId: l.id, layerIndex: i, input: prev, output: cur, preActivation: Z });
      } else { cache.push({ layerId: l.id, layerIndex: i, input: prev, output: prev }); }
    }
    return { output: cur[0], cache };
  }
  computeLoss(pred: number[], label: number, fn: string): { loss: number; outputGradient: number[] } {
    if (fn === 'crossEntropy') { const p = Math.max(pred[label], 1e-15); return { loss: -Math.log(p), outputGradient: ceGrad(pred, label) }; }
    if (fn === 'mse') { const t = vec(pred.length, 0); t[label] = 1; return { loss: pred.reduce((s, p, i) => s + (p - t[i]) ** 2, 0) / pred.length, outputGradient: mseGrad(pred, t) }; }
    return this.computeLoss(pred, label, 'crossEntropy');
  }
  backwardPass(layers: { id: number; type: string; params: Record<string, any> }[], cache: ModeEForwardCacheEntry[], outGrad: number[]): ModeELayerGradient[] {
    const grads: ModeELayerGradient[] = []; let dZ = [outGrad];
    for (let i = layers.length - 1; i >= 0; i--) { const l = layers[i], c = cache[i]; if (!c) { grads.unshift({ layerId: l.id, layerType: l.type, gradientNorm: 0, gradientStats: { min: 0, max: 0, mean: 0, std: 0 } }); continue; }
      if (l.type === 'output' || l.type === 'dense') { if (!l.params['weights']) l.params['weights'] = randMat(l.params['units'], c.input[0].length, 0.1); const W = l.params['weights'] as number[][]; const act = l.params['activation'] ?? 'relu'; const dzl = actDeriv(dZ, c, act, l.type === 'output');
        const dW = dot(transpose(dzl), c.input), db = dzl[0], dA = dot(dzl, W); const flat = ([] as number[]).concat(...dW); const gs = stats(flat);
        grads.unshift({ layerId: l.id, layerType: l.type, weightGradients: dW, biasGradients: db, inputGradient: dA, gradientNorm: l2Norm(flat), gradientStats: gs }); dZ = dA; continue; }
      if (l.type === 'activation') { const t = l.params['activationType'] ?? 'relu'; let dA: number[][] = dZ; if (t === 'relu') dA = hadamard(dZ, reluDeriv(c.input)); else if (t === 'sigmoid') dA = hadamard(dZ, sigmoidDeriv(c.output)); else if (t === 'tanh') dA = hadamard(dZ, tanhDeriv(c.output)); const f = ([] as number[]).concat(...dA);
        grads.unshift({ layerId: l.id, layerType: l.type, inputGradient: dA, gradientNorm: l2Norm(f), gradientStats: stats(f) }); dZ = dA; continue; }
      grads.unshift({ layerId: l.id, layerType: l.type, inputGradient: dZ, gradientNorm: 0, gradientStats: { min: 0, max: 0, mean: 0, std: 0 } });
    } return grads;
  }
  applyGradients(layers: { id: number; type: string; params: Record<string, any> }[], grads: ModeELayerGradient[], config: ModeETrainingConfig, cache: ModeEForwardCacheEntry[]): ModeEParameterSnapshot[] {
    const snaps: ModeEParameterSnapshot[] = []; const lr = config.learningRate; this.incOptT++;
    for (let i = 0; i < layers.length; i++) { const l = layers[i], g = grads.find(x => x.layerId === l.id); if (!g || !g.weightGradients) { snaps.push({ layerId: l.id }); continue; }
      const W = (l.params['weights'] as number[][]) ?? randMat(l.params['units'], cache[i]?.input[0]?.length ?? 1, 0.1);
      const b = (l.params['bias'] as number[]) ?? randVec(l.params['units'], 0);
      const wb = W.map(r => [...r]), bb = [...b];
      if (!this.incOptM[l.id]) { this.incOptM[l.id] = { w: mat(W.length, W[0].length), b: vec(b.length) }; this.incOptV[l.id] = { w: mat(W.length, W[0].length), b: vec(b.length) }; this.incOptVel[l.id] = { w: mat(W.length, W[0].length), b: vec(b.length) }; }
      applyOpt(W, g.weightGradients!, this.incOptVel[l.id].w, this.incOptV[l.id].w, b, g.biasGradients!, this.incOptVel[l.id].b, this.incOptV[l.id].b, config.optimizer, lr, this.incOptT);
      l.params['weights'] = W; l.params['bias'] = b;
      snaps.push({ layerId: l.id, weightsBefore: wb, weightsAfter: W.map(r => [...r]), biasBefore: bb, biasAfter: [...b], weightChange: subMat(W.map(r => [...r]), wb).map(r => r.map(v => Math.abs(v))), biasChange: b.map((v, j) => Math.abs(v - bb[j])) });
    } return snaps;
  }
  trainingStep(layers: { id: number; type: string; params: Record<string, any> }[], input: number[], label: number, config: ModeETrainingConfig, iteration: number): ModeEBackpropStep {
    const { output, cache } = this.forwardPass(layers, input);
    const { loss, outputGradient } = this.computeLoss(output, label, config.lossFunction);
    const grads = this.backwardPass(layers, cache, outputGradient);
    const snaps = this.applyGradients(layers, grads, config, cache);
    return { iteration, phase: 'update', layerIndex: layers.length - 1, totalLayers: layers.length, forwardCache: cache, loss, predictedClass: output.indexOf(Math.max(...output)), trueClass: label, predictions: output, layerGradients: grads, parameterSnapshots: snaps, optimizerState: { t: this.incOptT } };
  }

  // ---- incremental step API ------------------------------------------------
  initIncStep(layers: { id: number; type: string; params: Record<string, any> }[], input: number[], label: number, config: ModeETrainingConfig): void {
    this.layers = layers; this.input = input; this.label = label; this.config = config;
    this.incCache = [];
    this.incGradients = [];
    this.incSnapshots = [];
    this.incBackwardDZ = [];
    this.incOutputGrad = [];
    this.incLoss = 0; this.incPredClass = 0; this.incTrueClass = label;
  }

  /** Forward pass for ONE layer pair (from layer fromIdx to layer toIdx) */
  stepForwardPair(fromIdx: number, toIdx: number): ModeEForwardCacheEntry | null {
    // Ensure Lazy init of weights
    const toLayer = this.layers[toIdx];
    if (toLayer.type !== 'dense' && toLayer.type !== 'output') {
      this.incCache.push({ layerId: toLayer.id, layerIndex: toIdx, input: this.incCache[fromIdx]?.output ?? [this.input], output: this.incCache[fromIdx]?.output ?? [this.input] });
      return this.incCache[this.incCache.length - 1];
    }
    const prev = toIdx === 1 ? [this.input] : (this.incCache[fromIdx]?.output ?? [this.input]);
    if (!toLayer.params['weights']) toLayer.params['weights'] = randMat(toLayer.params['units'], prev[0].length, 0.1);
    if (!toLayer.params['bias']) toLayer.params['bias'] = randVec(toLayer.params['units'], 0);
    const W = toLayer.params['weights'] as number[][];
    const b = toLayer.params['bias'] as number[];
    const Z = addVecToRows(dot(prev, transpose(W)), b);
    const act = toLayer.params['activation'] ?? 'relu';
    const A = activate(Z, act, toLayer.type === 'output');
    const entry: ModeEForwardCacheEntry = { layerId: toLayer.id, layerIndex: toIdx, input: prev, output: A, preActivation: Z };
    this.incCache.push(entry);
    // Also push a pass-through entry for fromIdx if not already cached
    if (fromIdx === 0 && this.incCache.length === 1) {
      this.incCache.unshift({ layerId: this.layers[0].id, layerIndex: 0, input: [this.input], output: [this.input] });
    }
    return entry;
  }

  /** Compute loss (requires full forward pass completed) */
  stepLoss(): { loss: number; outputGradient: number[] } | null {
    const lastCache = this.incCache[this.incCache.length - 1];
    if (!lastCache?.output?.[0]) return null;
    const pred = lastCache.output[0];
    const { loss, outputGradient } = this.computeLoss(pred, this.label, this.config.lossFunction);
    this.incOutputGrad = outputGradient;
    this.incLoss = loss;
    this.incPredClass = pred.indexOf(Math.max(...pred));
    this.incPredictions = pred;
    this.incBackwardDZ = [outputGradient]; // start backward propagation
    return { loss, outputGradient };
  }

  /** Backward pass for ONE layer pair (from toIdx backward to fromIdx) */
  stepBackwardPair(toIdx: number): ModeELayerGradient | null {
    const layer = this.layers[toIdx];
    const cache = this.incCache[toIdx];
    if (!cache) return null;
    if (layer.type === 'dense' || layer.type === 'output') {
      if (!layer.params['weights']) layer.params['weights'] = randMat(layer.params['units'], cache.input[0].length, 0.1);
      const W = layer.params['weights'] as number[][];
      const act = layer.params['activation'] ?? 'relu';
      const dzl = actDeriv(this.incBackwardDZ, cache, act, layer.type === 'output');
      const dW = dot(transpose(dzl), cache.input);
      const db = dzl[0];
      const dA = dot(dzl, W);
      const flat = ([] as number[]).concat(...dW);
      const g: ModeELayerGradient = { layerId: layer.id, layerType: layer.type, weightGradients: dW, biasGradients: db, inputGradient: dA, gradientNorm: l2Norm(flat), gradientStats: stats(flat) };
      this.incGradients.unshift(g);
      this.incBackwardDZ = dA;
      return g;
    }
    const g0: ModeELayerGradient = { layerId: layer.id, layerType: layer.type, gradientNorm: 0, gradientStats: { min: 0, max: 0, mean: 0, std: 0 } };
    this.incGradients.unshift(g0);
    return g0;
  }

  /** Apply optimizer to ONE layer's stored gradient */
  stepUpdate(layerIdx: number): ModeEParameterSnapshot | null {
    const layer = this.layers[layerIdx];
    if (layer.type !== 'dense' && layer.type !== 'output') return null;
    const grad = this.incGradients.find(g => g.layerId === layer.id);
    if (!grad?.weightGradients) return null;
    const ci = this.incCache[layerIdx];
    const W = (layer.params['weights'] as number[][]) ?? randMat(layer.params['units'], ci?.input[0]?.length ?? 1, 0.1);
    const b = (layer.params['bias'] as number[]) ?? randVec(layer.params['units'], 0);
    const wb = W.map(r => [...r]), bb = [...b];
    if (!this.incOptM[layer.id]) { this.incOptM[layer.id] = { w: mat(W.length, W[0].length), b: vec(b.length) }; this.incOptV[layer.id] = { w: mat(W.length, W[0].length), b: vec(b.length) }; this.incOptVel[layer.id] = { w: mat(W.length, W[0].length), b: vec(b.length) }; }
    this.incOptT++;
    applyOpt(W, grad.weightGradients!, this.incOptVel[layer.id].w, this.incOptV[layer.id].w, b, grad.biasGradients!, this.incOptVel[layer.id].b, this.incOptV[layer.id].b, this.config.optimizer, this.config.learningRate, this.incOptT);
    layer.params['weights'] = W; layer.params['bias'] = b;
    const sn: ModeEParameterSnapshot = { layerId: layer.id, weightsBefore: wb, weightsAfter: W.map(r => [...r]), biasBefore: bb, biasAfter: [...b], weightChange: subMat(W.map(r => [...r]), wb).map(r => r.map(v => Math.abs(v))), biasChange: b.map((v, j) => Math.abs(v - bb[j])) };
    this.incSnapshots.push(sn);
    return sn;
  }

  /** Build current incremental step result for UI display */
  buildIncStep(phase: ModeEBackpropPhase, layerIdx: number): ModeEBackpropStep {
    return { iteration: 0, phase, layerIndex: layerIdx, totalLayers: this.layers.length, forwardCache: [...this.incCache],
      loss: this.incLoss, predictedClass: this.incPredClass, trueClass: this.incTrueClass, predictions: this.incPredictions,
      layerGradients: [...this.incGradients], parameterSnapshots: [...this.incSnapshots] };
  }

  // ---- decision boundary / reset / dataset generators (unchanged) ----------
  computeDecisionBoundary(layers: { id: number; type: string; params: Record<string, any> }[], res: number, xR: [number, number], yR: [number, number]): { resolution: number; xMin: number; xMax: number; yMin: number; yMax: number; grid: number[][] } {
    const grid: number[][] = []; const [xMin, xMax] = xR; const [yMin, yMax] = yR; const dx = (xMax - xMin) / (res - 1); const dy = (yMax - yMin) / (res - 1);
    for (let yi = 0; yi < res; yi++) { const row: number[] = []; const y = yMin + yi * dy; for (let xi = 0; xi < res; xi++) { const { output } = this.forwardPass(layers, [xMin + xi * dx, y]); row.push(output.indexOf(Math.max(...output))); } grid.push(row); } return { resolution: res, xMin, xMax, yMin, yMax, grid };
  }
  reset(): void { this.incOptT = 0; this.incOptM = {}; this.incOptV = {}; this.incOptVel = {}; }
  static generateXorData(n = 200, noise = 0.05): ModeEDatasetSample[] { const s: ModeEDatasetSample[] = []; const h = Math.floor(n / 4); const cs: [number, number, number][] = [[0, 0, 0], [1, 1, 0], [0, 1, 1], [1, 0, 1]]; for (const [cx, cy, l] of cs) for (let i = 0; i < h; i++) s.push({ input: [cx + (Math.random() - 0.5) * 2 * noise, cy + (Math.random() - 0.5) * 2 * noise], label: l }); return s; }
  static generateCircleData(n = 300, noise = 0.1): ModeEDatasetSample[] { const s: ModeEDatasetSample[] = []; const h = Math.floor(n / 2); for (let i = 0; i < h; i++) { const a = Math.random() * 2 * Math.PI; const r = 0.3 + (Math.random() - 0.5) * 2 * noise; s.push({ input: [0.5 + r * Math.cos(a), 0.5 + r * Math.sin(a)], label: 0 }); } for (let i = 0; i < h; i++) { const a = Math.random() * 2 * Math.PI; const r = 0.65 + (Math.random() - 0.5) * 2 * noise; s.push({ input: [0.5 + r * Math.cos(a), 0.5 + r * Math.sin(a)], label: 1 }); } return s; }
  static generateSpiralData(n = 400, classes = 3, noise = 0.2): ModeEDatasetSample[] { const s: ModeEDatasetSample[] = []; const pc = Math.floor(n / classes); for (let c = 0; c < classes; c++) for (let i = 0; i < pc; i++) { const t = (i / pc) * 2.5 * Math.PI; const r = t / (2.5 * Math.PI); const a = t + (c * 2 * Math.PI) / classes; s.push({ input: [0.5 + r * Math.cos(a) * 0.45 + (Math.random() - 0.5) * noise, 0.5 + r * Math.sin(a) * 0.45 + (Math.random() - 0.5) * noise], label: c }); } return s; }
  static generateBlobData(n = 400, centers: [number, number][] = [[0.3, 0.3], [0.7, 0.7], [0.3, 0.7]], noise = 0.08): ModeEDatasetSample[] { const s: ModeEDatasetSample[] = []; const pb = Math.floor(n / centers.length); for (let c = 0; c < centers.length; c++) { const [cx, cy] = centers[c]; for (let i = 0; i < pb; i++) s.push({ input: [cx + (Math.random() - 0.5) * 2 * noise, cy + (Math.random() - 0.5) * 2 * noise], label: c }); } return s; }
}
