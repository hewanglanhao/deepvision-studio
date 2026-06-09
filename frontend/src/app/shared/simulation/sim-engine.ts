import {
  ColorMode,
  Connection,
  DataSample,
  ExperimentResult,
  ForwardInputAsset,
  ForwardTensor,
  InputPreprocessConfig,
  LayerDraft,
  LayerType,
  MetricPoint,
  ModelTemplate,
  NetworkLayer,
  OptimizerType,
  PresetTask,
  SchedulerType,
  TensorShape,
  TrainingDataInfo,
  TrainingState
} from './sim-models';

export class SimEngine {
  private static readonly maxVisualizationSide = 56;
  private static readonly edgeKernel3x3: number[][] = [
    [-1, -1, -1],
    [-1, 8, -1],
    [-1, -1, -1]
  ];

  private static readonly blurKernel3x3: number[][] = [
    [1 / 16, 2 / 16, 1 / 16],
    [2 / 16, 4 / 16, 2 / 16],
    [1 / 16, 2 / 16, 1 / 16]
  ];

  /** 定义 A 模式可选网络模板，把 MLP、CNN、Residual CNN 等结构表示成可编辑的层序列。 */
  static templates(): ModelTemplate[] {
    const inputDraft = (name = 'Input'): LayerDraft => ({
      type: 'input',
      name,
      inputs: [],
      params: {
        inputKind: 'image',
        width: 32,
        height: 32,
        channels: 3,
        featureCount: 4,
        colorMode: 'rgb',
        preprocessing: {
          resizeMode: 'fit',
          targetWidth: 32,
          targetHeight: 32,
          colorMode: 'rgb',
          normalize: 'zero-one',
          invert: false
        }
      }
    });
    const tableInputDraft = (name = 'CSV Input', featureCount = 8): LayerDraft => ({
      type: 'input',
      name,
      inputs: [],
      params: {
        inputKind: 'table',
        width: 1,
        height: 1,
        channels: 1,
        featureCount,
        colorMode: 'grayscale',
        preprocessing: {
          resizeMode: 'none',
          targetWidth: 1,
          targetHeight: 1,
          colorMode: 'original',
          normalize: 'zero-one',
          invert: false
        }
      }
    });

    return [
      {
        id: 'csv-mlp',
        name: 'CSV / Tabular MLP',
        description: 'CSV Input -> Dense -> Dropout -> Dense -> Output',
        layers: [
          tableInputDraft(),
          { type: 'dense', name: 'Dense 1', inputs: [], params: { units: 64, activation: 'relu' } },
          { type: 'dropout', name: 'Dropout', inputs: [], params: { rate: 0.2, training: false } },
          { type: 'dense', name: 'Dense 2', inputs: [], params: { units: 32, activation: 'relu' } },
          { type: 'output', name: 'Output', inputs: [], params: { units: 2, activation: 'softmax' } }
        ]
      },
      {
        id: 'binary-mlp',
        name: 'Binary Classification MLP',
        description: '2-feature table input -> Dense -> Dense -> 2-class output',
        layers: [
          tableInputDraft('2D Point Input', 2),
          { type: 'dense', name: 'Dense 1', inputs: [], params: { units: 32, activation: 'relu' } },
          { type: 'dropout', name: 'Dropout', inputs: [], params: { rate: 0.1, training: false } },
          { type: 'dense', name: 'Dense 2', inputs: [], params: { units: 16, activation: 'relu' } },
          { type: 'output', name: 'Binary Output', inputs: [], params: { units: 2, activation: 'softmax' } }
        ]
      },
      {
        id: 'regression-mlp',
        name: 'Regression MLP',
        description: 'Numeric table input -> Dense -> Dense -> 1-value output',
        layers: [
          tableInputDraft('Regression Input', 5),
          { type: 'dense', name: 'Dense 1', inputs: [], params: { units: 64, activation: 'relu' } },
          { type: 'dense', name: 'Dense 2', inputs: [], params: { units: 32, activation: 'relu' } },
          { type: 'output', name: 'Value Output', inputs: [], params: { units: 1, activation: 'none' } }
        ]
      },
      {
        id: 'mlp-basic',
        name: 'MLP Basic',
        description: 'Input -> Flatten -> Dense -> Output',
        layers: [
          inputDraft(),
          { type: 'flatten', name: 'Flatten', inputs: [], params: {} },
          { type: 'dense', name: 'Dense 1', inputs: [], params: { units: 128, activation: 'relu' } },
          { type: 'output', name: 'Output', inputs: [], params: { units: 10, activation: 'softmax' } }
        ]
      },
      {
        id: 'cnn-classic',
        name: 'CNN Classic',
        description: 'Conv -> Pool -> Conv -> Flatten -> Dense -> Output',
        layers: [
          inputDraft(),
          {
            type: 'conv2d',
            name: 'Conv 1',
            inputs: [],
            params: {
              outChannels: 8,
              kernelSize: 3,
              stride: 1,
              padding: 1,
              dilation: 1,
              kernelMatrix: SimEngine.edgeKernel3x3.map((row) => [...row]),
              activation: 'relu'
            }
          },
          {
            type: 'pool2d',
            name: 'Pool 1',
            inputs: [],
            params: {
              mode: 'max',
              kernelSize: 2,
              stride: 2,
              padding: 0
            }
          },
          {
            type: 'conv2d',
            name: 'Conv 2',
            inputs: [],
            params: {
              outChannels: 16,
              kernelSize: 3,
              stride: 1,
              padding: 1,
              dilation: 1,
              kernelMatrix: SimEngine.blurKernel3x3.map((row) => [...row]),
              activation: 'relu'
            }
          },
          { type: 'flatten', name: 'Flatten', inputs: [], params: {} },
          { type: 'dense', name: 'Dense 1', inputs: [], params: { units: 64, activation: 'relu' } },
          { type: 'output', name: 'Output', inputs: [], params: { units: 10, activation: 'softmax' } }
        ]
      },
      {
        id: 'residual-cnn',
        name: 'Residual CNN',
        description: 'Conv stem -> Residual Block -> Pool -> Residual Block -> Output',
        layers: [
          inputDraft('Image Input'),
          {
            type: 'conv2d',
            name: 'Stem Conv',
            inputs: [],
            params: {
              outChannels: 16,
              kernelSize: 3,
              stride: 1,
              padding: 1,
              dilation: 1,
              kernelMatrix: SimEngine.edgeKernel3x3.map((row) => [...row]),
              activation: 'relu'
            }
          },
          {
            type: 'residual',
            name: 'Residual Block 1',
            inputs: [],
            params: {
              outChannels: 16,
              kernelSize: 3,
              stride: 1,
              padding: 1,
              activation: 'relu',
              useProjection: false
            }
          },
          {
            type: 'pool2d',
            name: 'Downsample',
            inputs: [],
            params: {
              mode: 'max',
              kernelSize: 2,
              stride: 2,
              padding: 0
            }
          },
          {
            type: 'residual',
            name: 'Residual Block 2',
            inputs: [],
            params: {
              outChannels: 32,
              kernelSize: 3,
              stride: 2,
              padding: 1,
              activation: 'relu',
              useProjection: true
            }
          },
          { type: 'flatten', name: 'Flatten', inputs: [], params: {} },
          { type: 'dense', name: 'Dense 1', inputs: [], params: { units: 96, activation: 'relu' } },
          { type: 'output', name: 'Output', inputs: [], params: { units: 10, activation: 'softmax' } }
        ]
      },
      {
        id: 'analyzer-lite',
        name: 'Analyzer Lite',
        description: 'Conv -> Activation -> Pool -> Flatten -> Output',
        layers: [
          inputDraft('Image Input'),
          {
            type: 'conv2d',
            name: 'Conv Edge',
            inputs: [],
            params: {
              outChannels: 4,
              kernelSize: 3,
              stride: 1,
              padding: 1,
              dilation: 1,
              kernelMatrix: SimEngine.edgeKernel3x3.map((row) => [...row]),
              activation: 'none'
            }
          },
          { type: 'activation', name: 'ReLU', inputs: [], params: { activationType: 'relu' } },
          {
            type: 'pool2d',
            name: 'Pool',
            inputs: [],
            params: {
              mode: 'avg',
              kernelSize: 2,
              stride: 2,
              padding: 0
            }
          },
          { type: 'flatten', name: 'Flatten', inputs: [], params: {} },
          { type: 'output', name: 'Output', inputs: [], params: { units: 10, activation: 'softmax' } }
        ]
      }
    ];
  }

  /** 生成演示数据集：MNIST 用单通道灰度张量，CIFAR 用三通道 RGB 张量，用来模拟不同输入形态。 */
  static generateDataset(count: number, mode: 'mnist' | 'cifar'): DataSample[];
  /** 兼容旧调用签名，仍然返回可送入输入层的演示样本。 */
  static generateDataset(count: number, _pixelCount: number, mode: 'mnist' | 'cifar'): DataSample[];
  /** 构造带像素值、shape 和预览灰度图的样本，供前端在无真实数据集时演示 forward 流程。 */
  static generateDataset(count: number, arg2: number | 'mnist' | 'cifar', arg3?: 'mnist' | 'cifar'): DataSample[] {
    const mode: 'mnist' | 'cifar' = typeof arg2 === 'number' ? (arg3 ?? 'mnist') : arg2;
    const width = mode === 'mnist' ? 28 : 32;
    const height = width;
    const channels = mode === 'mnist' ? 1 : 3;
    const total = width * height * channels;

    return Array.from({ length: count }, (_, idx) => {
      const phase = idx * 0.19;
      const values = Array.from({ length: total }, (_, i) => {
        const y = Math.floor(i / (width * channels));
        const x = Math.floor((i % (width * channels)) / channels);
        const c = i % channels;
        const wave = Math.sin((x + 1) * 0.17 + (y + 1) * 0.09 + phase + c * 0.8) * 0.24 + 0.5;
        const noise = Math.cos((x + y * 1.7 + c * 2.3 + idx) * 0.11) * 0.13;
        return Math.max(0, Math.min(1, wave + noise));
      });

      const previewPixels = channels === 1
        ? values.slice()
        : SimEngine.projectRgbToGray(values, width, height);

      return {
        id: idx + 1,
        label: idx % 10,
        pixels: values,
        width,
        height,
        channels,
        colorMode: channels === 1 ? 'grayscale' : 'rgb',
        previewPixels
      };
    });
  }

  /** 把内置样本包装成 forward 输入资产：保留原始张量，同时生成真正送入网络的预处理张量。 */
  static createForwardInputAssetFromSample(sample: DataSample, preprocess: InputPreprocessConfig): ForwardInputAsset {
    const originalTensor: ForwardTensor = {
      kind: 'tensor3d',
      shape: [sample.height, sample.width, sample.channels],
      values: sample.pixels.slice(),
      colorMode: sample.colorMode
    };
    const prepared = SimEngine.prepareInputTensor(originalTensor, preprocess);

    return {
      id: `sample-${sample.id}`,
      source: 'dataset',
      name: `Sample #${sample.id}`,
      originalWidth: sample.width,
      originalHeight: sample.height,
      originalChannels: sample.channels,
      originalColorMode: sample.colorMode,
      originalTensor,
      prepared,
      label: `${sample.label}`
    };
  }

  /** 把上传/本地图像转成 [H, W, C] 张量，并按输入层配置完成 resize、颜色通道和归一化。 */
  static createForwardInputAssetFromImageData(params: {
    id: string;
    name: string;
    source: 'dataset' | 'upload';
    imageData: ImageData;
    preprocess: InputPreprocessConfig;
    previewUrl?: string;
    label?: string;
  }): ForwardInputAsset {
    const { imageData, preprocess } = params;
    const original = SimEngine.imageDataToRgbTensor(imageData);
    const prepared = SimEngine.prepareInputTensor(original, preprocess);
    const originalChannels = original.shape[2] ?? 3;
    const originalColorMode = originalChannels === 1 ? 'grayscale' : 'rgb';

    return {
      id: params.id,
      source: params.source,
      name: params.name,
      previewUrl: params.previewUrl,
      originalWidth: imageData.width,
      originalHeight: imageData.height,
      originalChannels,
      originalColorMode,
      originalTensor: original,
      prepared,
      label: params.label
    };
  }

  /** 重建顺序网络连接，让前一层输出自然成为后一层输入，符合 A 模式线性 forward 路径。 */
  static rebuildLinearConnections(layers: NetworkLayer[]): Connection[] {
    if (layers.length < 2) {
      return [];
    }
    return layers.slice(0, -1).map((layer, idx) => ({ from: layer.id, to: layers[idx + 1].id }));
  }

  /** 把张量 shape 格式化成 [H, W, C] 或 [N] 文本，便于观察每层尺寸变化。 */
  static formatShapeLabel(shape: TensorShape): string {
    if (shape.length === 0) {
      return '[]';
    }
    return `[${shape.join(', ')}]`;
  }

  /** 前端快速推导输出 shape：卷积改变空间尺寸和通道数，池化降采样，Flatten 拉平成向量。 */
  static inferLayerOutputShape(layer: NetworkLayer, inputShapes: TensorShape[]): TensorShape {
    const inputShape = inputShapes[0] ?? [];
    if (layer.type === 'input') {
      if (layer.params.inputKind === 'table') {
        return [Math.max(1, layer.params.featureCount ?? 1)];
      }
      return [layer.params.height, layer.params.width, layer.params.channels];
    }
    if (layer.type === 'conv2d') {
      if (inputShape.length !== 3) {
        return [];
      }
      const [h, w] = inputShape;
      const k = Math.max(1, layer.params.kernelSize);
      const s = Math.max(1, layer.params.stride);
      const p = Math.max(0, layer.params.padding);
      const d = Math.max(1, layer.params.dilation);
      const effectiveK = d * (k - 1) + 1;
      const outH = Math.floor((h + p * 2 - effectiveK) / s) + 1;
      const outW = Math.floor((w + p * 2 - effectiveK) / s) + 1;
      return outH > 0 && outW > 0 ? [outH, outW, Math.max(1, layer.params.outChannels)] : [];
    }
    if (layer.type === 'pool2d') {
      if (inputShape.length !== 3) {
        return [];
      }
      const [h, w, c] = inputShape;
      const k = Math.max(1, layer.params.kernelSize);
      const s = Math.max(1, layer.params.stride);
      const p = Math.max(0, layer.params.padding);
      const outH = Math.floor((h + p * 2 - k) / s) + 1;
      const outW = Math.floor((w + p * 2 - k) / s) + 1;
      return outH > 0 && outW > 0 ? [outH, outW, c] : [];
    }
    if (layer.type === 'residual') {
      if (inputShape.length !== 3) {
        return [];
      }
      const [h, w] = inputShape;
      const k = Math.max(1, layer.params.kernelSize);
      const s = Math.max(1, layer.params.stride);
      const p = Math.max(0, layer.params.padding);
      const midH = Math.floor((h + p * 2 - k) / s) + 1;
      const midW = Math.floor((w + p * 2 - k) / s) + 1;
      const outH = Math.floor((midH + p * 2 - k) / 1) + 1;
      const outW = Math.floor((midW + p * 2 - k) / 1) + 1;
      return midH > 0 && midW > 0 && outH > 0 && outW > 0
        ? [outH, outW, Math.max(1, layer.params.outChannels)]
        : [];
    }
    if (layer.type === 'flatten') {
      return [SimEngine.shapeElementCount(inputShape)];
    }
    if (layer.type === 'dense') {
      return [Math.max(1, layer.params.units)];
    }
    if (layer.type === 'activation' || layer.type === 'dropout') {
      return inputShape;
    }
    return [Math.max(1, layer.params.units)];
  }

  /** 估算卷积核、残差投影、Dense 权重和 bias 数量，用来说明模型容量和计算规模。 */
  static parameterCount(layers: NetworkLayer[], connections: Connection[] = SimEngine.rebuildLinearConnections(layers)): number {
    const shapeById = new Map<number, TensorShape>();
    const orderedLayers = connections.length > 0
      ? layers
      : layers.slice().sort((a, b) => a.id - b.id);

    for (const layer of orderedLayers) {
      const inputShapes = (layer.inputs ?? [])
        .map((id) => shapeById.get(id))
        .filter((shape): shape is TensorShape => shape !== undefined);
      shapeById.set(layer.id, SimEngine.inferLayerOutputShape(layer, inputShapes));
    }

    let total = 0;
    for (const layer of layers) {
      if (layer.type === 'conv2d') {
        const inputShape = shapeById.get(layer.inputs[0] ?? -1);
        const inC = inputShape && inputShape.length === 3 ? inputShape[2] : 1;
        const k = Math.max(1, layer.params.kernelSize);
        total += k * k * inC * Math.max(1, layer.params.outChannels);
        total += Math.max(1, layer.params.outChannels);
      }
      if (layer.type === 'residual') {
        const inputShape = shapeById.get(layer.inputs[0] ?? -1);
        const inC = inputShape && inputShape.length === 3 ? inputShape[2] : 1;
        const outC = Math.max(1, layer.params.outChannels);
        const k = Math.max(1, layer.params.kernelSize);
        total += k * k * inC * outC + outC;
        total += k * k * outC * outC + outC;
        if (layer.params.useProjection) {
          total += inC * outC + outC;
        }
      }
      if (layer.type === 'dense' || layer.type === 'output') {
        const inputShape = shapeById.get(layer.inputs[0] ?? -1);
        const inDim = Math.max(1, SimEngine.shapeElementCount(inputShape ?? []));
        const outDim = Math.max(1, layer.params.units);
        total += inDim * outDim + outDim;
      }
    }
    return total;
  }

  /** 将层类型转成界面标签，帮助区分输入、卷积、池化、Flatten、Dense 和输出层。 */
  static layerTypeLabel(type: LayerType): string {
    const map: Record<LayerType, string> = {
      input: 'Input',
      conv2d: 'Conv2D',
      pool2d: 'Pool2D',
      residual: 'Residual',
      flatten: 'Flatten',
      dense: 'Dense',
      activation: 'Activation',
      dropout: 'Dropout',
      output: 'Output'
    };
    return map[type];
  }

  /** 根据学习率调度策略生成下一轮学习率，用于训练模式的曲线演示。 */
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

  /** 生成一轮训练指标点，模拟 loss、accuracy、gradient norm 随 epoch 变化的趋势。 */
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
    const overfitGap = Math.max(0.015, valGap * 1.8);
    const valLoss = Math.max(0.04, loss + overfitGap + Math.sin(currentEpoch * 0.43) * 0.015);
    const gradientNorm = Math.max(0.0004, 1.2 * Math.exp(-1.9 * progress) + Math.abs(Math.sin(currentEpoch * 0.5)) * 0.08 + lrPenalty * 2);
    const weightMean = Math.sin(currentEpoch * 0.2) * 0.018;
    const weightStd = Math.max(0.02, 0.16 - progress * 0.06 + depthBonus * 0.08);
    const elapsedSeconds = currentEpoch * 2.8;
    const etaSeconds = Math.max(0, (totalEpochs - currentEpoch) * 2.8);

    return {
      step: currentEpoch,
      loss,
      valLoss,
      accuracy: trainAcc,
      valAccuracy: valAcc,
      lr: lrNow,
      gradientNorm,
      weightMean,
      weightStd,
      elapsedSeconds,
      etaSeconds
    };
  }

  /** 把训练历史指标转换成 SVG 折线点，用于展示 loss/accuracy/lr 曲线。 */
  static buildPolyline(history: MetricPoint[], metric: 'loss' | 'valLoss' | 'accuracy' | 'valAccuracy' | 'lr' | 'gradientNorm'): string {
    if (history.length === 0) {
      return '';
    }

    const width = 280;
    const height = 120;
    const maxStep = Math.max(1, history[history.length - 1].step);
    const values = history.map((point) => point[metric]);
    const isLoss = metric === 'loss' || metric === 'valLoss';
    const maxValue = Math.max(...values, isLoss ? 1.8 : metric === 'lr' ? Math.max(...values, 0.001) : 1);
    const minValue = Math.min(...values, isLoss ? 0.02 : 0);
    const span = Math.max(0.001, maxValue - minValue);

    return history
      .map((point) => {
        const x = (point.step / maxStep) * width;
        const y = height - ((point[metric] - minValue) / span) * height;
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(' ');
  }

  /** 用模型深度、优化器和训练轮数估算任务表现，支撑训练对比面板的教学演示。 */
  static evaluateTask(task: PresetTask, layers: NetworkLayer[], optimizer: OptimizerType, totalEpochs: number): number {
    const depthFactor = layers.length / 7;
    const optFactor = optimizer === 'AdamW' ? 1.08 : optimizer === 'Adam' ? 1.05 : optimizer === 'SGD' ? 0.9 : 0.98;
    const base = task.type === 'classification' ? 0.72 : 0.84;
    return Math.min(0.985, Math.max(0.45, base + depthFactor * 0.12 * optFactor + totalEpochs * 0.0012));
  }

  /** 模拟“加深网络/换激活函数/换优化器”的实验结果，说明结构和训练策略会影响精度与速度。 */
  static runExperiment(mode: 'deeper' | 'activation' | 'optimizer', baseline: number, totalEpochs: number): ExperimentResult {
    let candidateAcc = baseline;
    let speed = 1;
    let label = 'Baseline';
    if (mode === 'deeper') {
      label = 'Increase Depth';
      candidateAcc = Math.min(0.99, baseline + 0.05);
      speed = 0.78;
    } else if (mode === 'activation') {
      label = 'Switch to GELU';
      candidateAcc = Math.min(0.99, baseline + 0.03);
      speed = 0.9;
    } else {
      label = 'Switch to AdamW';
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

  /** 生成教学用特征图、Grad-CAM 和层激活预览，帮助观察训练过程中网络关注区域的变化。 */
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
    const seedValues = sample.previewPixels.length > 0 ? sample.previewPixels : sample.pixels;

    const featureMaps = Array.from({ length: mapCount }, (_, mapIdx) => {
      return Array.from({ length: matrixSize * matrixSize }, (_, i) => {
        const base = seedValues[i % seedValues.length];
        const signal = Math.sin(i * 0.33 + mapIdx * 0.8 + currentEpoch * 0.17) * 0.18;
        return Math.max(0, Math.min(1, base + signal));
      });
    });

    const gradCamMap = Array.from({ length: 100 }, (_, i) => {
      const source = seedValues[i % seedValues.length];
      const focus = Math.sin(i * 0.21 + currentEpoch * 0.25) * 0.25 + 0.5;
      return Math.max(0, Math.min(1, source * 0.6 + focus * 0.4));
    });

    const base = seedValues.reduce((sum, value) => sum + value, 0) / Math.max(1, seedValues.length);
    const inferenceActivations = layers.map((layer, idx) => {
      const units = SimEngine.layerUnits(layer);
      const length = Math.min(16, Math.max(4, Math.floor(units / 8)));
      const values = Array.from({ length }, (_, i) => {
        const wave = Math.sin(base * 10 + i * 0.6 + idx * 0.7) * 0.35 + 0.5;
        return Math.max(0, Math.min(1, wave));
      });
      return { layerName: layer.name, values };
    });

    return { featureMaps, gradCamMap, inferenceActivations };
  }

  /** 生成混淆矩阵示例，用来解释分类模型哪些类别容易被互相误判。 */
  static buildConfusionMatrix(seed: number, classes = 10): number[][] {
    return Array.from({ length: classes }, (_, i) => {
      return Array.from({ length: classes }, (_, j) => {
        if (i === j) {
          return Math.round(72 + Math.abs(Math.sin(seed * 0.2 + i)) * 26);
        }
        return Math.round(Math.abs(Math.cos(seed * 0.32 + i * 0.5 + j * 0.9)) * 12);
      });
    });
  }

  /** 生成 loss landscape 示例，直观说明优化器和模型深度会影响损失曲面的形态。 */
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

  /** 初始化训练状态，给 loss、accuracy、梯度范数等曲线提供起点。 */
  static createInitialTrainingState(learningRate: number): TrainingState {
    return {
      status: 'idle',
      currentEpoch: 0,
      currentLr: learningRate,
      latestLoss: 1.7,
      latestValLoss: 1.78,
      latestAccuracy: 0.22,
      latestValAccuracy: 0.2,
      latestGradientNorm: 1.2,
      latestWeightMean: 0,
      latestWeightStd: 0.16,
      elapsedSeconds: 0,
      etaSeconds: 0
    };
  }

  /** 检查监督训练是否具备样本和标签；没有标签就无法计算分类损失和准确率。 */
  static canRunSupervisedTraining(info: TrainingDataInfo): { ok: boolean; message: string } {
    if (info.sampleCount <= 0) {
      return { ok: false, message: 'No training samples available.' };
    }
    if (!info.hasLabels) {
      return { ok: false, message: 'Training mode requires labels.' };
    }
    return { ok: true, message: '' };
  }

  /** 推进一轮训练模拟，把新的学习率、loss、accuracy 和梯度范数写回训练状态。 */
  static nextTrainingState(params: {
    state: TrainingState;
    totalEpochs: number;
    layers: NetworkLayer[];
    optimizer: OptimizerType;
    learningRate: number;
    scheduler: SchedulerType;
    lrDecay: number;
  }): { state: TrainingState; metric: MetricPoint } {
    const nextEpoch = params.state.currentEpoch + 1;
    const point = SimEngine.pushMetricPoint({
      currentEpoch: nextEpoch,
      totalEpochs: params.totalEpochs,
      layers: params.layers,
      optimizer: params.optimizer,
      learningRate: params.learningRate,
      scheduler: params.scheduler,
      lrDecay: params.lrDecay
    });

    return {
      state: {
        status: 'running',
        currentEpoch: nextEpoch,
        currentLr: point.lr,
        latestLoss: point.loss,
        latestAccuracy: point.accuracy,
        latestValAccuracy: point.valAccuracy,
        latestValLoss: point.valLoss,
        latestGradientNorm: point.gradientNorm,
        latestWeightMean: point.weightMean,
        latestWeightStd: point.weightStd,
        elapsedSeconds: point.elapsedSeconds,
        etaSeconds: point.etaSeconds
      },
      metric: point
    };
  }

  /** 把 0-1 响应值映射成灰度或热力颜色，用于特征图、Grad-CAM 和矩阵可视化。 */
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

  /** 执行输入预处理，保证送入 Python forward 的张量尺寸、颜色通道和数值范围与输入层一致。 */
  private static prepareInputTensor(original: ForwardTensor, preprocess: InputPreprocessConfig): ForwardInputAsset['prepared'] {
    if (original.shape.length !== 3) {
      return {
        tensor: original,
        displayTensor: original,
        notes: ['Input tensor is not image-like.']
      };
    }

    let tensor = SimEngine.cloneTensor(original);
    const notes: string[] = [];

    if (preprocess.colorMode !== 'original') {
      tensor = SimEngine.convertColorMode(tensor, preprocess.colorMode);
      notes.push(`color=${preprocess.colorMode}`);
    }

    if (preprocess.resizeMode === 'fit' && preprocess.targetWidth && preprocess.targetHeight) {
      const targetW = Math.max(1, Math.floor(preprocess.targetWidth));
      const targetH = Math.max(1, Math.floor(preprocess.targetHeight));
      if (tensor.shape[0] !== targetH || tensor.shape[1] !== targetW) {
        tensor = SimEngine.resizeTensorNearest(tensor, targetW, targetH);
        notes.push(`resize=${targetW}x${targetH}`);
      }
    }

    if (preprocess.invert) {
      tensor = {
        ...tensor,
        values: tensor.values.map((value) => 1 - value)
      };
      notes.push('invert=true');
    }

    if (preprocess.normalize === 'zero-one') {
      tensor = {
        ...tensor,
        values: SimEngine.normalizeValues(tensor.values)
      };
      notes.push('normalize=zero-one');
    }

    const displayValues = tensor.shape.length === 3 && tensor.shape[2] === 3
      ? tensor.values.slice()
      : SimEngine.normalizeValues(tensor.values);

    return {
      tensor,
      displayTensor: {
        ...tensor,
        values: displayValues,
        colorMode: tensor.shape.length === 3 && tensor.shape[2] === 3 ? 'rgb' : tensor.colorMode
      },
      notes
    };
  }

  /** 把浏览器 ImageData 展开成 CNN 常用的 [height, width, channels] 张量，并识别是否为灰度图。 */
  private static imageDataToRgbTensor(imageData: ImageData): ForwardTensor {
    const values = new Array(imageData.width * imageData.height * 3);
    let isGray = true;
    for (let y = 0; y < imageData.height; y += 1) {
      for (let x = 0; x < imageData.width; x += 1) {
        const sourceIndex = (y * imageData.width + x) * 4;
        const targetIndex = (y * imageData.width + x) * 3;
        const r = imageData.data[sourceIndex] / 255;
        const g = imageData.data[sourceIndex + 1] / 255;
        const b = imageData.data[sourceIndex + 2] / 255;
        if (isGray && (Math.abs(r - g) > 1e-6 || Math.abs(g - b) > 1e-6)) {
          isGray = false;
        }
        values[targetIndex] = r;
        values[targetIndex + 1] = g;
        values[targetIndex + 2] = b;
      }
    }

    if (isGray) {
      const gray = new Array(imageData.width * imageData.height);
      for (let i = 0; i < gray.length; i += 1) {
        gray[i] = values[i * 3] ?? 0;
      }
      return {
        kind: 'tensor3d',
        shape: [imageData.height, imageData.width, 1],
        values: gray,
        colorMode: 'grayscale'
      };
    }

    return {
      kind: 'tensor3d',
      shape: [imageData.height, imageData.width, 3],
      values,
      colorMode: 'rgb'
    };
  }

  /** 转换 RGB/灰度通道数；通道数会直接决定第一层卷积核需要覆盖的输入深度。 */
  private static convertColorMode(tensor: ForwardTensor, colorMode: ColorMode): ForwardTensor {
    if (tensor.shape.length !== 3) {
      return tensor;
    }
    const [h, w, c] = tensor.shape;
    if (colorMode === 'grayscale') {
      if (c === 1) {
        return { ...tensor, colorMode: 'grayscale' };
      }
      const gray = SimEngine.projectRgbToGray(tensor.values, w, h, c);
      return {
        kind: 'tensor3d',
        shape: [h, w, 1],
        values: gray,
        colorMode: 'grayscale'
      };
    }

    if (c >= 3) {
      const trimmed = new Array(h * w * 3);
      for (let y = 0; y < h; y += 1) {
        for (let x = 0; x < w; x += 1) {
          const src = (y * w + x) * c;
          const dst = (y * w + x) * 3;
          trimmed[dst] = tensor.values[src] ?? 0;
          trimmed[dst + 1] = tensor.values[src + 1] ?? 0;
          trimmed[dst + 2] = tensor.values[src + 2] ?? 0;
        }
      }
      return {
        ...tensor,
        shape: [h, w, 3],
        values: trimmed,
        colorMode: 'rgb'
      };
    }
    const rgb = new Array(h * w * 3);
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const base = (y * w + x) * c;
        const v = tensor.values[base];
        const target = (y * w + x) * 3;
        rgb[target] = v;
        rgb[target + 1] = v;
        rgb[target + 2] = v;
      }
    }
    return {
      kind: 'tensor3d',
      shape: [h, w, 3],
      values: rgb,
      colorMode: 'rgb'
    };
  }

  /** 用最近邻采样调整输入张量尺寸，让任意上传图片都能适配固定大小的网络输入层。 */
  private static resizeTensorNearest(tensor: ForwardTensor, width: number, height: number): ForwardTensor {
    if (tensor.shape.length !== 3) {
      return tensor;
    }
    const [srcH, srcW, c] = tensor.shape;
    const out = new Array(width * height * c);

    for (let y = 0; y < height; y += 1) {
      const srcY = Math.min(srcH - 1, Math.floor((y / height) * srcH));
      for (let x = 0; x < width; x += 1) {
        const srcX = Math.min(srcW - 1, Math.floor((x / width) * srcW));
        for (let ch = 0; ch < c; ch += 1) {
          const srcIndex = ((srcY * srcW) + srcX) * c + ch;
          const targetIndex = ((y * width) + x) * c + ch;
          out[targetIndex] = tensor.values[srcIndex];
        }
      }
    }

    return {
      ...tensor,
      shape: [height, width, c],
      values: out
    };
  }

  /** 将张量数值缩放到 0-1，避免原始像素尺度影响卷积和 Dense 的响应大小。 */
  private static normalizeValues(values: number[]): number[] {
    if (values.length === 0) {
      return [];
    }
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    for (const value of values) {
      if (value < min) min = value;
      if (value > max) max = value;
    }
    const span = Math.max(1e-6, max - min);
    return values.map((value) => (value - min) / span);
  }

  /** 按亮度权重把 RGB 投影到灰度单通道，便于演示单通道图像如何进入 CNN。 */
  private static projectRgbToGray(values: number[], width: number, height: number, channels = 3): number[] {
    const out = new Array(width * height);
    for (let y = 0; y < height; y += 1) {
      for (let x = 0; x < width; x += 1) {
        const base = ((y * width) + x) * channels;
        const r = values[base] ?? 0;
        const g = values[base + Math.min(1, channels - 1)] ?? 0;
        const b = values[base + Math.min(2, channels - 1)] ?? 0;
        out[y * width + x] = r * 0.299 + g * 0.587 + b * 0.114;
      }
    }
    return out;
  }

  /** 复制张量的 shape、values 和 labels，避免预处理或可视化步骤修改原始输入。 */
  private static cloneTensor(tensor: ForwardTensor): ForwardTensor {
    return {
      ...tensor,
      shape: [...tensor.shape] as TensorShape,
      values: tensor.values.slice(),
      labels: tensor.labels ? [...tensor.labels] : undefined
    };
  }

  /** 计算 shape 包含的元素总数；Flatten 输出长度和 Dense 输入维度都依赖它。 */
  private static shapeElementCount(shape: TensorShape): number {
    if (shape.length === 0) {
      return 0;
    }
    return shape.reduce((acc, value) => acc * value, 1);
  }

  /** 估算某层可展示的单元数量：卷积看通道数，Dense/Output 看神经元或类别数。 */
  private static layerUnits(layer: NetworkLayer): number {
    if (layer.type === 'input') {
      if (layer.params.inputKind === 'table') {
        return Math.max(1, layer.params.featureCount ?? 1);
      }
      return layer.params.width * layer.params.height * layer.params.channels;
    }
    if (layer.type === 'conv2d') {
      return layer.params.outChannels;
    }
    if (layer.type === 'pool2d') {
      return layer.params.kernelSize;
    }
    if (layer.type === 'residual') {
      return layer.params.outChannels;
    }
    if (layer.type === 'dense') {
      return layer.params.units;
    }
    if (layer.type === 'output') {
      return layer.params.units;
    }
    return 16;
  }
}
