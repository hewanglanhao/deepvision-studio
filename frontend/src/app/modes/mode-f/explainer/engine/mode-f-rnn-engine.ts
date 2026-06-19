// Pure TS RNN engine — simple tanh RNN + BPTT
import type { ModeFForwardResult, ModeFRnnGradient, ModeFRnnState, ModeFSequenceSample, ModeFStepResult, ModeFTrainingConfig } from '../models/mode-f.types';

function vec(n: number, fill = 0): number[] { return Array(n).fill(fill); }
function mat(r: number, c: number, fill = 0): number[][] { return Array.from({ length: r }, () => vec(c, fill)); }
function randMat(r: number, c: number, scale = 0.08): number[][] { return mat(r, c).map(row => row.map(() => (Math.random() - 0.5) * 2 * scale)); }
function randVec(n: number, scale = 0.08): number[] { return vec(n).map(() => (Math.random() - 0.5) * 2 * scale); }

function dot(A: number[][], B: number[][]): number[][] {
  const C = mat(A.length, B[0].length);
  for (let i = 0; i < A.length; i++) for (let j = 0; j < B[0].length; j++) for (let k = 0; k < B.length; k++) C[i][j] += A[i][k] * B[k][j];
  return C;
}
function dotVec(A: number[][], b: number[]): number[] {
  const r = vec(A.length);
  for (let i = 0; i < A.length; i++) for (let j = 0; j < b.length; j++) r[i] += A[i][j] * b[j];
  return r;
}
function outer(a: number[], b: number[]): number[][] {
  const M = mat(a.length, b.length);
  for (let i = 0; i < a.length; i++) for (let j = 0; j < b.length; j++) M[i][j] = a[i] * b[j];
  return M;
}
function add(a: number[], b: number[]): number[] { return a.map((v, i) => v + b[i]); }
function sub(a: number[], b: number[]): number[] { return a.map((v, i) => v - b[i]); }
function scale(a: number[], s: number): number[] { return a.map(v => v * s); }
function hprod(a: number[], b: number[]): number[] { return a.map((v, i) => v * b[i]); }

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / sum);
}

export class ModeFRnnEngine {
  // weights
  Wxh: number[][]; Whh: number[][]; Why: number[][] = [];
  bh: number[]; by: number[] = [];
  // optimizer state
  oWxh: number[][] = []; oWhh: number[][] = []; oWhy: number[][] = [];
  obh: number[] = []; oby: number[] = [];
  ovWxh: number[][] = []; ovWhh: number[][] = []; ovWhy: number[][] = [];
  ovbh: number[] = []; ovby: number[] = [];
  private t = 0;

  constructor(public inputDim: number, public hiddenDim: number, public outputDim: number) {
    this.Wxh = randMat(hiddenDim, inputDim);
    this.Whh = randMat(hiddenDim, hiddenDim);
    this.Why = randMat(outputDim, hiddenDim);
    this.bh = randVec(hiddenDim, 0);
    this.by = randVec(outputDim, 0);
  }

  reset(): void {
    this.Wxh = randMat(this.hiddenDim, this.inputDim);
    this.Whh = randMat(this.hiddenDim, this.hiddenDim);
    this.Why = randMat(this.outputDim, this.hiddenDim);
    this.bh = randVec(this.hiddenDim, 0);
    this.by = randVec(this.outputDim, 0);
    this.oWxh = []; this.oWhh = []; this.oWhy = []; this.obh = []; this.oby = [];
    this.ovWxh = []; this.ovWhh = []; this.ovWhy = []; this.ovbh = []; this.ovby = [];
    this.t = 0;
  }

  /** Return a snapshot of all current weight values */
  getWeights(): { Wxh: number[][]; Whh: number[][]; Why: number[][]; bh: number[]; by: number[] } {
    const copyMat = (m: number[][]) => m.map(r => [...r]);
    return { Wxh: copyMat(this.Wxh), Whh: copyMat(this.Whh), Why: copyMat(this.Why), bh: [...this.bh], by: [...this.by] };
  }

  forward(sequence: number[][]): ModeFForwardResult {
    const T = sequence.length;
    const h = vec(this.hiddenDim, 0); // h_0 = 0
    const states: ModeFRnnState[] = [];
    const predictions: number[][] = [];

    for (let t = 0; t < T; t++) {
      const x = sequence[t];
      const whh_h = dotVec(this.Whh, h);
      const wxh_x = dotVec(this.Wxh, x);
      const hNext = add(add(wxh_x, whh_h), this.bh).map(Math.tanh);
      const logits = add(dotVec(this.Why, hNext), this.by);
      const probs = softmax(logits);
      states.push({ hidden: [...hNext], output: [...probs] });
      predictions.push([...probs]);
      h.length = 0; h.push(...hNext);
    }

    return { states, predictions, finalPrediction: predictions[predictions.length - 1] };
  }

  trainStep(sample: ModeFSequenceSample, config: ModeFTrainingConfig, iteration: number): ModeFStepResult {
    const T = sample.inputs.length;
    const result = this.forward(sample.inputs);

    // Cross-entropy at last step
    const probs = result.finalPrediction;
    const loss = -Math.log(Math.max(probs[sample.label], 1e-15));

    // BPTT
    const dhnext = vec(this.hiddenDim, 0);
    let dLdWhy = mat(this.outputDim, this.hiddenDim, 0);
    let dLdWhh = mat(this.hiddenDim, this.hiddenDim, 0);
    let dLdWxh = mat(this.hiddenDim, this.inputDim, 0);
    let dLdbh = vec(this.hiddenDim, 0);
    let dLdby = vec(this.outputDim, 0);

    // Output gradient at last step
    const dy = [...probs]; dy[sample.label] -= 1;
    for (let i = 0; i < this.outputDim; i++) {
      dLdby[i] += dy[i];
      for (let j = 0; j < this.hiddenDim; j++) {
        dLdWhy[i][j] += dy[i] * result.states[T - 1].hidden[j];
      }
    }

    let dh = vec(this.hiddenDim, 0);
    for (let i = 0; i < this.outputDim; i++) {
      for (let j = 0; j < this.hiddenDim; j++) {
        dh[j] += dy[i] * this.Why[i][j];
      }
    }
    dh = add(dh, dhnext);

    // Backward through time
    for (let t = T - 1; t >= 0; t--) {
      const h = result.states[t].hidden;
      const x = sample.inputs[t];
      const hPrev = t > 0 ? result.states[t - 1].hidden : vec(this.hiddenDim, 0);

      // dh raw / d(tanh) -> (1 - tanh^2) = (1 - h^2)
      const dtanh = h.map(v => 1 - v * v);
      dh = hprod(dh, dtanh);

      for (let i = 0; i < this.hiddenDim; i++) {
        dLdbh[i] += dh[i];
        for (let j = 0; j < this.inputDim; j++) dLdWxh[i][j] += dh[i] * x[j];
        for (let j = 0; j < this.hiddenDim; j++) dLdWhh[i][j] += dh[i] * hPrev[j];
      }

      // Propagate to previous step
      const dhNew = vec(this.hiddenDim, 0);
      for (let i = 0; i < this.hiddenDim; i++) {
        for (let j = 0; j < this.hiddenDim; j++) {
          dhNew[j] += dh[i] * this.Whh[i][j];
        }
      }
      dh = dhNew;
    }

    const gradNorm = (() => {
      let s = 0;
      for (const r of dLdWhh) for (const v of r) s += v * v;
      for (const r of dLdWxh) for (const v of r) s += v * v;
      return Math.sqrt(s);
    })();

    const gradient: ModeFRnnGradient = { dWhy: dLdWhy, dWhh: dLdWhh, dWxh: dLdWxh, dbh: dLdbh, dby: dLdby, gradientNorm: gradNorm };

    // Snapshot weights before update
    const wBefore = this.getWeights();

    // Apply gradients
    this.t++;
    this.applyUpdate(this.Wxh, dLdWxh, this.oWxh, this.ovWxh, config);
    this.applyUpdate(this.Whh, dLdWhh, this.oWhh, this.ovWhh, config);
    this.applyUpdate(this.Why, dLdWhy, this.oWhy, this.ovWhy, config);
    this.applyVecUpdate(this.bh, dLdbh, this.obh, this.ovbh, config);
    this.applyVecUpdate(this.by, dLdby, this.oby, this.ovby, config);

    // Snapshot weights after update
    const wAfter = this.getWeights();

    return {
      iteration, loss, predictedClass: probs.indexOf(Math.max(...probs)), trueClass: sample.label,
      forwardResult: result, gradient, hiddenDim: this.hiddenDim, timeSteps: T, outputProbs: probs,
      weightSnapshot: {
        WxhBefore: wBefore.Wxh, WxhAfter: wAfter.Wxh,
        WhhBefore: wBefore.Whh, WhhAfter: wAfter.Whh,
        WhyBefore: wBefore.Why, WhyAfter: wAfter.Why,
        bhBefore: wBefore.bh, bhAfter: wAfter.bh,
        byBefore: wBefore.by, byAfter: wAfter.by,
      },
    };
  }

  private applyUpdate(W: number[][], dW: number[][], o: number[][], v: number[][], c: ModeFTrainingConfig): void {
    const lr = c.learningRate;
    if (o.length === 0) { for (let i = 0; i < W.length; i++) { o[i] = vec(W[i].length); v[i] = vec(W[i].length); } }
    if (c.optimizer === 'adam') {
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      for (let i = 0; i < W.length; i++) {
        for (let j = 0; j < W[i].length; j++) {
          o[i][j] = b1 * o[i][j] + (1 - b1) * dW[i][j];
          v[i][j] = b2 * v[i][j] + (1 - b2) * dW[i][j] * dW[i][j];
          const mh = o[i][j] / (1 - Math.pow(b1, this.t));
          const vh = v[i][j] / (1 - Math.pow(b2, this.t));
          W[i][j] -= lr * mh / (Math.sqrt(vh) + eps);
        }
      }
    } else if (c.optimizer === 'momentum') {
      const beta = 0.9;
      for (let i = 0; i < W.length; i++) for (let j = 0; j < W[i].length; j++) { o[i][j] = beta * o[i][j] + lr * dW[i][j]; W[i][j] -= o[i][j]; }
    } else {
      for (let i = 0; i < W.length; i++) for (let j = 0; j < W[i].length; j++) W[i][j] -= lr * dW[i][j];
    }
  }

  private applyVecUpdate(w: number[], dw: number[], o: number[], v: number[], c: ModeFTrainingConfig): void {
    const lr = c.learningRate;
    if (o.length === 0) { o.length = w.length; o.fill(0); v.length = w.length; v.fill(0); }
    if (c.optimizer === 'adam') {
      const b1 = 0.9, b2 = 0.999, eps = 1e-8;
      for (let i = 0; i < w.length; i++) {
        o[i] = b1 * o[i] + (1 - b1) * dw[i]; v[i] = b2 * v[i] + (1 - b2) * dw[i] * dw[i];
        const mh = o[i] / (1 - Math.pow(b1, this.t)); const vh = v[i] / (1 - Math.pow(b2, this.t));
        w[i] -= lr * mh / (Math.sqrt(vh) + eps);
      }
    } else if (c.optimizer === 'momentum') {
      const beta = 0.9;
      for (let i = 0; i < w.length; i++) { o[i] = beta * o[i] + lr * dw[i]; w[i] -= o[i]; }
    } else {
      for (let i = 0; i < w.length; i++) w[i] -= lr * dw[i];
    }
  }
}
