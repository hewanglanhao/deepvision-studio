import {
  Connection,
  DataSample,
  ExperimentResult,
  LayerType,
  MetricPoint,
  ModelTemplate,
  NetworkLayer,
  OptimizerType,
  PresetTask,
  SchedulerType
} from './sim-models';

export class SimEngine {
  static readonly kernelPresets: Record<string, number[][]> = {
    edge: [
      [-1, -1, -1],
      [-1, 8, -1],
      [-1, -1, -1]
    ],
    sharpen: [
      [0, -1, 0],
      [-1, 5, -1],
      [0, -1, 0]
    ],
    blur: [
      [1 / 16, 2 / 16, 1 / 16],
      [2 / 16, 4 / 16, 2 / 16],
      [1 / 16, 2 / 16, 1 / 16]
    ],
    emboss: [
      [-2, -1, 0],
      [-1, 1, 1],
      [0, 1, 2]
    ]
  };

  static templates(): ModelTemplate[] {
    return [
      {
        id: 'mlp-basic',
        name: 'MLP 基线',
        description: '输入-全连接-全连接-输出，适合基础分类演示。',
        layers: [
          { type: 'input', name: 'Input', units: 784, kernelSize: 0, activation: 'none', dropoutRate: 0 },
          { type: 'dense', name: 'Dense 1', units: 128, kernelSize: 0, activation: 'relu', dropoutRate: 0 },
          { type: 'dense', name: 'Dense 2', units: 64, kernelSize: 0, activation: 'relu', dropoutRate: 0 },
          { type: 'output', name: 'Output', units: 10, kernelSize: 0, activation: 'softmax', dropoutRate: 0 }
        ]
      },
      {
        id: 'cnn-classic',
        name: 'CNN 经典',
        description: '卷积+池化+全连接，贴近手写数字/小图分类场景。',
        layers: [
          { type: 'input', name: 'Input', units: 784, kernelSize: 0, activation: 'none', dropoutRate: 0 },
          { type: 'conv', name: 'Conv 1', units: 16, kernelSize: 3, activation: 'relu', dropoutRate: 0 },
          { type: 'pool', name: 'Pool 1', units: 16, kernelSize: 2, activation: 'none', dropoutRate: 0 },
          { type: 'conv', name: 'Conv 2', units: 32, kernelSize: 3, activation: 'relu', dropoutRate: 0 },
          { type: 'dense', name: 'Dense 1', units: 64, kernelSize: 0, activation: 'relu', dropoutRate: 0 },
          { type: 'output', name: 'Output', units: 10, kernelSize: 0, activation: 'softmax', dropoutRate: 0 }
        ]
      },
      {
        id: 'resnet-mini',
        name: 'ResNet Mini',
        description: '用于展示“残差思想”的简化结构。',
        layers: [
          { type: 'input', name: 'Input', units: 784, kernelSize: 0, activation: 'none', dropoutRate: 0 },
          { type: 'conv', name: 'Conv Stem', units: 16, kernelSize: 3, activation: 'relu', dropoutRate: 0 },
          { type: 'conv', name: 'Res Block A', units: 16, kernelSize: 3, activation: 'relu', dropoutRate: 0 },
          { type: 'conv', name: 'Res Block B', units: 16, kernelSize: 3, activation: 'relu', dropoutRate: 0 },
          { type: 'pool', name: 'Global Pool', units: 16, kernelSize: 2, activation: 'none', dropoutRate: 0 },
          { type: 'output', name: 'Output', units: 10, kernelSize: 0, activation: 'softmax', dropoutRate: 0 }
        ]
      },
      {
        id: 'lstm-lite',
        name: 'LSTM Lite',
        description: '以前馈形式模拟时序门控层，做教学占位。',
        layers: [
          { type: 'input', name: 'Input', units: 128, kernelSize: 0, activation: 'none', dropoutRate: 0 },
          { type: 'dense', name: 'Gate Dense', units: 64, kernelSize: 0, activation: 'tanh', dropoutRate: 0 },
          { type: 'dropout', name: 'Gate Dropout', units: 64, kernelSize: 0, activation: 'none', dropoutRate: 0.2 },
          { type: 'dense', name: 'Memory Dense', units: 64, kernelSize: 0, activation: 'tanh', dropoutRate: 0 },
          { type: 'output', name: 'Output', units: 10, kernelSize: 0, activation: 'softmax', dropoutRate: 0 }
        ]
      }
    ];
  }

  static generateDataset(count: number, pixelCount: number, mode: 'mnist' | 'cifar'): DataSample[] {
    return Array.from({ length: count }, (_, idx) => {
      const center = (idx + 1) / (count + 1);
      const pixels = Array.from({ length: pixelCount }, (_, i) => {
        const wave = Math.sin((i + 1) * 0.23 + idx * 0.8) * 0.22 + center;
        const noise = Math.cos((i + idx * 3) * 0.17) * (mode === 'mnist' ? 0.18 : 0.26);
        return Math.max(0, Math.min(1, wave + noise));
      });
      return {
        id: idx + 1,
        label: idx % 10,
        pixels
      };
    });
  }

  static rebuildLinearConnections(layers: NetworkLayer[]): Connection[] {
    const connections: Connection[] = [];
    for (let i = 0; i < layers.length - 1; i += 1) {
      connections.push({ from: layers[i].id, to: layers[i + 1].id });
    }
    return connections;
  }

  static parameterCount(layers: NetworkLayer[]): number {
    let count = 0;
    for (let i = 1; i < layers.length; i += 1) {
      const prev = layers[i - 1];
      const curr = layers[i];
      if (curr.type === 'dense' || curr.type === 'output') {
        count += prev.units * curr.units + curr.units;
      }
      if (curr.type === 'conv') {
        const kernel = curr.kernelSize || 3;
        count += kernel * kernel * Math.max(1, Math.floor(prev.units / 64)) * curr.units;
      }
    }
    return count;
  }

  static layerTypeLabel(type: LayerType): string {
    const map: Record<LayerType, string> = {
      input: '输入层',
      dense: '全连接',
      conv: '卷积层',
      pool: '池化层',
      dropout: 'Dropout',
      output: '输出层'
    };
    return map[type];
  }

  static nextLr(baseLr: number, scheduler: SchedulerType, decay: number, epoch: number, totalEpochs: number): number {
    if (scheduler === 'none') {
      return baseLr;
    }
    if (scheduler === 'step') {
      const phase = Math.floor(epoch / Math.max(1, totalEpochs / 5));
      return baseLr * Math.pow(decay, phase);
    }

    const cosine = 0.5 * (1 + Math.cos((Math.PI * epoch) / Math.max(1, totalEpochs)));
    return Math.max(baseLr * 0.1, baseLr * cosine);
  }

  static pushMetricPoint(params: {
    currentEpoch: number;
    totalEpochs: number;
    layers: NetworkLayer[];
    optimizer: OptimizerType;
    learningRate: number;
    scheduler: SchedulerType;
    lrDecay: number;
  }): MetricPoint {
    const { currentEpoch, totalEpochs, layers, optimizer, learningRate, scheduler, lrDecay } = params;

    const optimizerBonus: Record<OptimizerType, number> = {
      Adam: 0.09,
      AdamW: 0.1,
      RMSProp: 0.06,
      SGD: 0.03,
      Momentum: 0.05,
      Nesterov: 0.06,
      Adagrad: 0.04,
      Adadelta: 0.045
    };

    const depthBonus = Math.min(0.24, layers.length * 0.02);
    const lrNow = SimEngine.nextLr(learningRate, scheduler, lrDecay, currentEpoch, totalEpochs);
    const lrPenalty = lrNow > 0.01 ? 0.12 : lrNow < 0.00035 ? 0.05 : 0;
    const progress = currentEpoch / Math.max(1, totalEpochs);
    const baseLoss = 1.6 * Math.exp(-2.3 * progress) + 0.12;
    const jitter = Math.sin(currentEpoch * 0.65) * 0.02;

    const loss = Math.max(0.03, baseLoss + jitter + lrPenalty - depthBonus * 0.28);
    const trainAcc = Math.max(
      0.05,
      Math.min(
        0.995,
        0.2 + (1 - Math.exp(-2.8 * progress)) * 0.72 + depthBonus + optimizerBonus[optimizer] - lrPenalty
      )
    );
    const valGap = 0.012 + Math.max(0, layers.length - 6) * 0.004;
    const valAcc = Math.max(0.04, Math.min(0.992, trainAcc - valGap + jitter * 0.3));

    return {
      step: currentEpoch,
      loss,
      accuracy: trainAcc,
      valAccuracy: valAcc,
      lr: lrNow
    };
  }

  static buildPolyline(history: MetricPoint[], metric: 'loss' | 'accuracy' | 'valAccuracy'): string {
    if (history.length === 0) {
      return '';
    }

    const width = 280;
    const height = 120;
    const maxStep = Math.max(1, history[history.length - 1].step);
    const values = history.map((point) => point[metric]);
    const maxValue = Math.max(...values, metric === 'loss' ? 1.8 : 1);
    const minValue = Math.min(...values, metric === 'loss' ? 0.02 : 0);
    const span = Math.max(0.001, maxValue - minValue);

    return history
      .map((point) => {
        const x = (point.step / maxStep) * width;
        const y = height - ((point[metric] - minValue) / span) * height;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }

  static evaluateTask(task: PresetTask, layers: NetworkLayer[], optimizer: OptimizerType, totalEpochs: number): number {
    const depthFactor = layers.length / 7;
    const optFactor = optimizer === 'AdamW' ? 1.08 : optimizer === 'Adam' ? 1.05 : optimizer === 'SGD' ? 0.9 : 0.98;
    const base = task.type === 'classification' ? 0.72 : 0.84;
    return Math.min(0.985, Math.max(0.45, base + depthFactor * 0.12 * optFactor + totalEpochs * 0.0012));
  }

  static runExperiment(mode: 'deeper' | 'activation' | 'optimizer', baseline: number, totalEpochs: number): ExperimentResult {
    let candidateAcc = baseline;
    let speed = 1;
    let label = '基线';

    if (mode === 'deeper') {
      label = '增加网络深度';
      candidateAcc = Math.min(0.99, baseline + 0.05);
      speed = 0.78;
    } else if (mode === 'activation') {
      label = '将激活函数切换为 GELU';
      candidateAcc = Math.min(0.99, baseline + 0.03);
      speed = 0.9;
    } else {
      label = '优化器切换到 AdamW';
      candidateAcc = Math.min(0.99, baseline + 0.04);
      speed = 0.95;
    }

    return {
      name: label,
      epochs: totalEpochs,
      finalAccuracy: candidateAcc,
      speedScore: speed
    };
  }

  static refreshVisuals(params: {
    sample: DataSample | undefined;
    selectedDataset: string;
    currentEpoch: number;
    layers: NetworkLayer[];
  }): { featureMaps: number[][]; gradCamMap: number[]; inferenceActivations: { layerName: string; values: number[] }[] } {
    const { sample, selectedDataset, currentEpoch, layers } = params;
    if (!sample) {
      return { featureMaps: [], gradCamMap: [], inferenceActivations: [] };
    }

    const matrixSize = selectedDataset === 'MNIST' ? 8 : 10;
    const mapCount = 4;

    const featureMaps = Array.from({ length: mapCount }, (_, mapIdx) => {
      return Array.from({ length: matrixSize * matrixSize }, (_, i) => {
        const base = sample.pixels[i % sample.pixels.length];
        const signal = Math.sin(i * 0.33 + mapIdx * 0.8 + currentEpoch * 0.17) * 0.18;
        return Math.max(0, Math.min(1, base + signal));
      });
    });

    const gradCamMap = Array.from({ length: 100 }, (_, i) => {
      const source = sample.pixels[i % sample.pixels.length];
      const focus = Math.sin(i * 0.21 + currentEpoch * 0.25) * 0.25 + 0.5;
      return Math.max(0, Math.min(1, source * 0.6 + focus * 0.4));
    });

    const base = sample.pixels.reduce((sum, value) => sum + value, 0) / sample.pixels.length;
    const inferenceActivations = layers.map((layer, idx) => {
      const length = Math.min(16, Math.max(4, Math.floor(layer.units / 8)));
      const values = Array.from({ length }, (_, i) => {
        const wave = Math.sin(base * 10 + i * 0.6 + idx * 0.7) * 0.35 + 0.5;
        return Math.max(0, Math.min(1, wave));
      });
      return { layerName: layer.name, values };
    });

    return { featureMaps, gradCamMap, inferenceActivations };
  }

  static buildConfusionMatrix(seed: number, classes = 10): number[][] {
    return Array.from({ length: classes }, (_, i) => {
      const row = Array.from({ length: classes }, (_, j) => {
        if (i === j) {
          return Math.round(72 + Math.abs(Math.sin(seed * 0.2 + i)) * 26);
        }
        return Math.round(Math.abs(Math.cos(seed * 0.32 + i * 0.5 + j * 0.9)) * 12);
      });
      return row;
    });
  }

  static buildLossLandscape(
    latestLoss: number,
    valAcc: number,
    layerCount: number,
    optimizer: OptimizerType,
    size = 18
  ): number[][] {
    const optimizerFactor: Record<OptimizerType, number> = {
      Adam: 0.8,
      AdamW: 0.75,
      RMSProp: 0.9,
      SGD: 1.1,
      Momentum: 0.98,
      Nesterov: 0.92,
      Adagrad: 1.0,
      Adadelta: 0.96
    };

    const baseDepth = Math.max(0.5, 1.2 - layerCount * 0.05);
    const centerX = (0.48 + (1 - valAcc) * 0.1) * size;
    const centerY = (0.5 + latestLoss * 0.05) * size;
    const sharpness = optimizerFactor[optimizer] * baseDepth;

    return Array.from({ length: size }, (_, y) => {
      return Array.from({ length: size }, (_, x) => {
        const dx = (x - centerX) / size;
        const dy = (y - centerY) / size;
        const bowl = dx * dx * 1.8 + dy * dy * 1.2;
        const ripple = Math.sin((x + 1) * 0.55) * Math.cos((y + 1) * 0.47) * 0.04;
        const lossSurface = (bowl * sharpness + 0.12 + ripple) / 0.9;
        return Math.max(0, Math.min(1, lossSurface));
      });
    });
  }

  static imageDataToGrayMatrix(imageData: ImageData, invert = true): number[][] {
    const { width, height, data } = imageData;
    const matrix: number[][] = Array.from({ length: height }, () => Array(width).fill(0));

    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const idx = (y * width + x) * 4;
        const r = data[idx];
        const g = data[idx + 1];
        const b = data[idx + 2];
        const gray = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
        matrix[y][x] = invert ? 1 - gray : gray;
      }
    }

    return matrix;
  }

  static flattenMatrix(matrix: number[][]): number[] {
    return matrix.flat();
  }

  static convolve2d(
    matrix: number[][],
    kernel: number[][],
    options: { stride?: number; padding?: 'same' | 'valid'; relu?: boolean } = {}
  ): number[][] {
    const stride = options.stride ?? 1;
    const padding = options.padding ?? 'same';
    const relu = options.relu ?? true;

    const h = matrix.length;
    const w = matrix[0]?.length ?? 0;
    const kh = kernel.length;
    const kw = kernel[0]?.length ?? 0;

    const padH = padding === 'same' ? Math.floor(kh / 2) : 0;
    const padW = padding === 'same' ? Math.floor(kw / 2) : 0;

    const outH = Math.floor((h + 2 * padH - kh) / stride) + 1;
    const outW = Math.floor((w + 2 * padW - kw) / stride) + 1;

    const output = Array.from({ length: outH }, () => Array(outW).fill(0));

    for (let oy = 0; oy < outH; oy += 1) {
      for (let ox = 0; ox < outW; ox += 1) {
        let sum = 0;
        for (let ky = 0; ky < kh; ky += 1) {
          for (let kx = 0; kx < kw; kx += 1) {
            const iy = oy * stride + ky - padH;
            const ix = ox * stride + kx - padW;
            if (iy >= 0 && iy < h && ix >= 0 && ix < w) {
              sum += matrix[iy][ix] * kernel[ky][kx];
            }
          }
        }
        output[oy][ox] = relu ? Math.max(0, sum) : sum;
      }
    }

    return output;
  }

  static maxPool2d(matrix: number[][], poolSize = 2, stride = 2): number[][] {
    const h = matrix.length;
    const w = matrix[0]?.length ?? 0;
    const outH = Math.floor((h - poolSize) / stride) + 1;
    const outW = Math.floor((w - poolSize) / stride) + 1;
    const output = Array.from({ length: outH }, () => Array(outW).fill(0));

    for (let oy = 0; oy < outH; oy += 1) {
      for (let ox = 0; ox < outW; ox += 1) {
        let maxVal = -Infinity;
        for (let py = 0; py < poolSize; py += 1) {
          for (let px = 0; px < poolSize; px += 1) {
            const iy = oy * stride + py;
            const ix = ox * stride + px;
            maxVal = Math.max(maxVal, matrix[iy][ix]);
          }
        }
        output[oy][ox] = maxVal;
      }
    }

    return output;
  }

  static normalizeMatrix(matrix: number[][]): number[][] {
    const flat = matrix.flat();
    const min = Math.min(...flat);
    const max = Math.max(...flat);
    const span = Math.max(1e-6, max - min);
    return matrix.map((row) => row.map((v) => (v - min) / span));
  }

  static cellColor(value: number, colorMode: 'mono' | 'heat' = 'mono'): string {
    const clipped = Math.max(0, Math.min(1, value));
    if (colorMode === 'heat') {
      const r = Math.round(235 * clipped + 20);
      const g = Math.round(130 * (1 - clipped) + 40);
      const b = Math.round(65 * (1 - clipped) + 30);
      return `rgb(${r}, ${g}, ${b})`;
    }
    const c = Math.round(clipped * 255);
    return `rgb(${c}, ${c}, ${c})`;
  }
}
