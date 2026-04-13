import {
  ActivationType,
  ColorMode,
  Connection,
  Conv2DLayer,
  DataSample,
  DenseLayer,
  ExperimentResult,
  ForwardInputAsset,
  ForwardLayerResult,
  ForwardPassResult,
  ForwardTensor,
  InputLayer,
  InputPreprocessConfig,
  LayerDraft,
  LayerType,
  LayerValidationIssue,
  MetricPoint,
  ModelTemplate,
  NetworkLayer,
  OptimizerType,
  OutputLayer,
  Pool2DLayer,
  PresetTask,
  SchedulerType,
  TensorShape,
  TensorStats,
  TrainingDataInfo,
  TrainingState
} from './sim-models';

interface ExecutionGraph {
  nodesById: Map<number, NetworkLayer>;
  inbound: Map<number, number[]>;
  outbound: Map<number, number[]>;
  errors: string[];
  warnings: string[];
}

interface OperatorResult {
  tensor: ForwardTensor;
  transitionNote: string;
  paramsSummary: string[];
}

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

  static templates(): ModelTemplate[] {
    const inputDraft = (name = 'Input'): LayerDraft => ({
      type: 'input',
      name,
      inputs: [],
      params: {
        width: 32,
        height: 32,
        channels: 3,
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

    return [
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

  static generateDataset(count: number, mode: 'mnist' | 'cifar'): DataSample[];
  static generateDataset(count: number, _pixelCount: number, mode: 'mnist' | 'cifar'): DataSample[];
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

  static rebuildLinearConnections(layers: NetworkLayer[]): Connection[] {
    if (layers.length < 2) {
      return [];
    }
    return layers.slice(0, -1).map((layer, idx) => ({ from: layer.id, to: layers[idx + 1].id }));
  }

  static formatShapeLabel(shape: TensorShape): string {
    if (shape.length === 0) {
      return '[]';
    }
    return `[${shape.join(', ')}]`;
  }

  static inferLayerOutputShape(layer: NetworkLayer, inputShapes: TensorShape[]): TensorShape {
    const inputShape = inputShapes[0] ?? [];
    if (layer.type === 'input') {
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

  static validateLayerParams(layer: NetworkLayer, inputShapes: TensorShape[]): LayerValidationIssue[] {
    const issues: LayerValidationIssue[] = [];
    const inputShape = inputShapes[0] ?? [];
    const issue = (severity: 'error' | 'warning', message: string, field?: string): void => {
      issues.push({ layerId: layer.id, layerName: layer.name, severity, message, field });
    };

    if (layer.enabled === false) {
      issue('warning', 'Layer is disabled.');
      return issues;
    }

    if (layer.type !== 'input' && inputShapes.length === 0) {
      issue('error', 'Layer has no input tensor.');
      return issues;
    }

    if (layer.type === 'conv2d' || layer.type === 'pool2d') {
      if (inputShape.length !== 3) {
        issue('error', 'Conv/Pool requires an image-like input shape [H, W, C].', 'inputShape');
      }
    }

    if (layer.type === 'conv2d') {
      if (layer.params.kernelSize <= 0) {
        issue('error', 'kernelSize must be > 0.', 'kernelSize');
      }
      if (layer.params.stride <= 0) {
        issue('error', 'stride must be > 0.', 'stride');
      }
      if (layer.params.outChannels <= 0) {
        issue('error', 'outChannels must be > 0.', 'outChannels');
      }
      const outShape = SimEngine.inferLayerOutputShape(layer, inputShapes);
      if (outShape.length === 0) {
        issue('error', 'Invalid output shape. Check kernel/stride/padding/dilation.', 'padding');
      }
    }

    if (layer.type === 'pool2d') {
      if (layer.params.kernelSize <= 0) {
        issue('error', 'kernelSize must be > 0.', 'kernelSize');
      }
      if (layer.params.stride <= 0) {
        issue('error', 'stride must be > 0.', 'stride');
      }
      const outShape = SimEngine.inferLayerOutputShape(layer, inputShapes);
      if (outShape.length === 0) {
        issue('error', 'Invalid pool output shape. Check kernel/stride/padding.', 'padding');
      }
    }

    if (layer.type === 'dense' || layer.type === 'output') {
      if (layer.params.units <= 0) {
        issue('error', 'units must be > 0.', 'units');
      }
      if (inputShape.length === 0) {
        issue('error', 'Dense/Output requires a non-empty input shape.', 'inputShape');
      }
    }

    if (layer.type === 'dropout') {
      if (layer.params.rate < 0 || layer.params.rate >= 1) {
        issue('error', 'dropout rate must be in [0, 1).', 'rate');
      }
    }

    return issues;
  }

  static buildExecutionGraph(layers: NetworkLayer[], connections: Connection[]): ExecutionGraph {
    const nodesById = new Map<number, NetworkLayer>();
    const inbound = new Map<number, number[]>();
    const outbound = new Map<number, number[]>();
    const errors: string[] = [];
    const warnings: string[] = [];

    for (const layer of layers) {
      if (nodesById.has(layer.id)) {
        errors.push(`Duplicate layer id: ${layer.id}.`);
        continue;
      }
      nodesById.set(layer.id, layer);
      inbound.set(layer.id, []);
      outbound.set(layer.id, []);
    }

    const edgeSet = new Set<string>();
    const addEdge = (from: number, to: number, source: string): void => {
      if (!nodesById.has(from) || !nodesById.has(to)) {
        errors.push(`Invalid edge ${from} -> ${to} from ${source}.`);
        return;
      }
      if (from === to) {
        errors.push(`Self loop is not allowed: ${from} -> ${to}.`);
        return;
      }
      const key = `${from}->${to}`;
      if (edgeSet.has(key)) {
        return;
      }
      edgeSet.add(key);
      inbound.get(to)?.push(from);
      outbound.get(from)?.push(to);
    };

    for (const layer of layers) {
      for (const inputId of layer.inputs) {
        addEdge(inputId, layer.id, `layer(${layer.id}).inputs`);
      }
    }
    for (const edge of connections) {
      addEdge(edge.from, edge.to, 'connections');
    }

    for (const layer of layers) {
      const hasInput = (inbound.get(layer.id)?.length ?? 0) > 0;
      if (layer.type !== 'input' && !hasInput) {
        warnings.push(`Layer "${layer.name}" has no inbound edge.`);
      }
    }

    return { nodesById, inbound, outbound, errors, warnings };
  }

  static topologicalSort(graph: ExecutionGraph): { order: number[]; errors: string[] } {
    const errors: string[] = [];
    const indegree = new Map<number, number>();
    for (const [id, arr] of graph.inbound) {
      indegree.set(id, arr.length);
    }

    const queue: number[] = [];
    for (const [id, degree] of indegree) {
      if (degree === 0) {
        queue.push(id);
      }
    }

    const order: number[] = [];
    while (queue.length > 0) {
      const current = queue.shift();
      if (current === undefined) {
        break;
      }
      order.push(current);
      const nextIds = graph.outbound.get(current) ?? [];
      for (const nextId of nextIds) {
        const nextDegree = (indegree.get(nextId) ?? 0) - 1;
        indegree.set(nextId, nextDegree);
        if (nextDegree === 0) {
          queue.push(nextId);
        }
      }
    }

    if (order.length !== graph.nodesById.size) {
      errors.push('Graph contains a cycle or disconnected invalid dependency chain.');
    }
    return { order, errors };
  }

  static executeForwardGraph(params: {
    layers: NetworkLayer[];
    connections: Connection[];
    inputAsset: ForwardInputAsset | null;
  }): ForwardPassResult {
    const errors: string[] = [];
    const warnings: string[] = [];
    const validationIssues: LayerValidationIssue[] = [];
    const layerResults: ForwardLayerResult[] = [];
    const layerShapeMap: Record<number, string> = {};

    if (!params.inputAsset) {
      return {
        executionOrder: [],
        layerResults: [],
        layerShapeMap,
        finalTensor: null,
        finalTopK: [],
        validationIssues,
        shapePath: [],
        errors: ['Missing input asset.'],
        warnings: [],
        resolvedLayers: params.layers
      };
    }

    const graph = SimEngine.buildExecutionGraph(params.layers, params.connections);
    errors.push(...graph.errors);
    warnings.push(...graph.warnings);
    if (errors.length > 0) {
      return {
        executionOrder: [],
        layerResults: [],
        layerShapeMap,
        finalTensor: null,
        finalTopK: [],
        validationIssues,
        shapePath: [],
        errors,
        warnings,
        resolvedLayers: params.layers
      };
    }

    const sorted = SimEngine.topologicalSort(graph);
    errors.push(...sorted.errors);
    if (errors.length > 0) {
      return {
        executionOrder: sorted.order,
        layerResults: [],
        layerShapeMap,
        finalTensor: null,
        finalTopK: [],
        validationIssues,
        shapePath: [],
        errors,
        warnings,
        resolvedLayers: params.layers
      };
    }

    const tensorByLayer = new Map<number, ForwardTensor>();
    for (const layerId of sorted.order) {
      const layer = graph.nodesById.get(layerId);
      if (!layer) {
        continue;
      }

      const parentIds = graph.inbound.get(layerId) ?? [];
      const parentTensors = parentIds
        .map((id) => tensorByLayer.get(id))
        .filter((item): item is ForwardTensor => item !== undefined);
      const inputShapes = parentTensors.map((item) => item.shape);
      const validation = SimEngine.validateLayerParams(layer, inputShapes);
      validationIssues.push(...validation);
      const layerWarnings = validation.filter((issue) => issue.severity === 'warning').map((issue) => issue.message);
      const layerErrors = validation.filter((issue) => issue.severity === 'error').map((issue) => issue.message);
      warnings.push(...layerWarnings.map((msg) => `${layer.name}: ${msg}`));

      if (layerErrors.length > 0) {
        errors.push(...layerErrors.map((msg) => `${layer.name}: ${msg}`));
        continue;
      }

      try {
        const opResult = SimEngine.executeOperator(layer, parentTensors, params.inputAsset);
        tensorByLayer.set(layerId, opResult.tensor);

        const outputShape = opResult.tensor.shape;
        const outputShapeLabel = SimEngine.formatShapeLabel(outputShape);
        const inputShapeLabel = inputShapes.length > 0
          ? inputShapes.map((shape) => SimEngine.formatShapeLabel(shape)).join(', ')
          : '[]';
        const stats = SimEngine.computeTensorStats(opResult.tensor);
        const visualization = SimEngine.buildLayerVisualization(opResult.tensor);
        const layerResult: ForwardLayerResult = {
          layerId: layer.id,
          layerName: layer.name,
          layerType: layer.type,
          inputShapes,
          outputShape,
          inputShapeLabel,
          outputShapeLabel,
          shapeLabel: outputShapeLabel,
          transitionNote: opResult.transitionNote,
          paramsSummary: opResult.paramsSummary,
          warnings: layerWarnings,
          tensor: opResult.tensor,
          visualization,
          stats
        };
        layerResults.push(layerResult);
        layerShapeMap[layer.id] = outputShapeLabel;
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Unknown operator error.';
        errors.push(`${layer.name}: ${message}`);
      }
    }

    const finalLayer = layerResults[layerResults.length - 1];
    const finalTensor = finalLayer?.tensor ?? null;

    return {
      executionOrder: sorted.order,
      layerResults,
      layerShapeMap,
      finalTensor,
      finalTopK: finalTensor ? SimEngine.computeTensorStats(finalTensor).topK : [],
      validationIssues,
      shapePath: layerResults.map((item) => `${item.layerName}: ${item.outputShapeLabel}`),
      errors,
      warnings,
      resolvedLayers: params.layers
    };
  }

  static parameterCount(layers: NetworkLayer[], connections: Connection[] = SimEngine.rebuildLinearConnections(layers)): number {
    const result = SimEngine.executeForwardGraph({
      layers,
      connections,
      inputAsset: SimEngine.mockInputFromLayers(layers)
    });
    const shapeById = new Map<number, TensorShape>();
    for (const item of result.layerResults) {
      shapeById.set(item.layerId, item.outputShape);
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
      if (layer.type === 'dense' || layer.type === 'output') {
        const inputShape = shapeById.get(layer.inputs[0] ?? -1);
        const inDim = Math.max(1, SimEngine.shapeElementCount(inputShape ?? []));
        const outDim = Math.max(1, layer.params.units);
        total += inDim * outDim + outDim;
      }
    }
    return total;
  }

  static layerTypeLabel(type: LayerType): string {
    const map: Record<LayerType, string> = {
      input: 'Input',
      conv2d: 'Conv2D',
      pool2d: 'Pool2D',
      flatten: 'Flatten',
      dense: 'Dense',
      activation: 'Activation',
      dropout: 'Dropout',
      output: 'Output'
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

  static createInitialTrainingState(learningRate: number): TrainingState {
    return {
      status: 'idle',
      currentEpoch: 0,
      currentLr: learningRate,
      latestLoss: 1.7,
      latestAccuracy: 0.22,
      latestValAccuracy: 0.2
    };
  }

  static canRunSupervisedTraining(info: TrainingDataInfo): { ok: boolean; message: string } {
    if (info.sampleCount <= 0) {
      return { ok: false, message: 'No training samples available.' };
    }
    if (!info.hasLabels) {
      return { ok: false, message: 'Training mode requires labels.' };
    }
    return { ok: true, message: '' };
  }

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
        latestValAccuracy: point.valAccuracy
      },
      metric: point
    };
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

  private static executeOperator(layer: NetworkLayer, inputs: ForwardTensor[], inputAsset: ForwardInputAsset): OperatorResult {
    switch (layer.type) {
      case 'input':
        return SimEngine.runInputOperator(layer, inputAsset.prepared.tensor);
      case 'conv2d':
        return SimEngine.runConv2DOperator(layer, inputs[0]);
      case 'pool2d':
        return SimEngine.runPool2DOperator(layer, inputs[0]);
      case 'flatten':
        return SimEngine.runFlattenOperator(inputs[0]);
      case 'dense':
        return SimEngine.runDenseOperator(layer, inputs[0]);
      case 'activation':
        return SimEngine.runActivationOperator(layer, inputs[0]);
      case 'dropout':
        return SimEngine.runDropoutOperator(layer, inputs[0]);
      case 'output':
        return SimEngine.runOutputOperator(layer, inputs[0]);
      default:
        throw new Error('Unsupported layer type.');
    }
  }

  private static runInputOperator(layer: InputLayer, inputTensor: ForwardTensor): OperatorResult {
    return {
      tensor: {
        ...inputTensor,
        shape: inputTensor.shape,
        kind: SimEngine.kindFromShape(inputTensor.shape)
      },
      transitionNote: 'Input tensor enters the graph.',
      paramsSummary: [
        `network input: ${layer.params.width}x${layer.params.height}x${layer.params.channels}`,
        `preprocess: resize=${layer.params.preprocessing.resizeMode}, color=${layer.params.preprocessing.colorMode}, normalize=${layer.params.preprocessing.normalize}`
      ]
    };
  }

  private static runConv2DOperator(layer: Conv2DLayer, inputTensor: ForwardTensor): OperatorResult {
    if (inputTensor.shape.length !== 3) {
      throw new Error('Conv2D expects [H, W, C] tensor.');
    }
    const [h, w, c] = inputTensor.shape;
    const p = layer.params;
    const k = Math.max(1, p.kernelSize);
    const stride = Math.max(1, p.stride);
    const pad = Math.max(0, p.padding);
    const dilation = Math.max(1, p.dilation);
    const outC = Math.max(1, p.outChannels);
    const effectiveK = dilation * (k - 1) + 1;
    const outH = Math.floor((h + pad * 2 - effectiveK) / stride) + 1;
    const outW = Math.floor((w + pad * 2 - effectiveK) / stride) + 1;
    if (outH <= 0 || outW <= 0) {
      throw new Error('Conv2D output shape is invalid.');
    }

    const output = new Array(outH * outW * outC).fill(0);
    for (let oc = 0; oc < outC; oc += 1) {
      const kernel3d = SimEngine.resolveKernel3d(layer, oc, c, k);
      const bias = layer.params.bias?.[oc] ?? 0;
      for (let oy = 0; oy < outH; oy += 1) {
        for (let ox = 0; ox < outW; ox += 1) {
          let acc = bias;
          for (let ic = 0; ic < c; ic += 1) {
            for (let ky = 0; ky < k; ky += 1) {
              for (let kx = 0; kx < k; kx += 1) {
                const iy = oy * stride + ky * dilation - pad;
                const ix = ox * stride + kx * dilation - pad;
                if (iy < 0 || iy >= h || ix < 0 || ix >= w) {
                  continue;
                }
                const inputValue = SimEngine.tensor3dGet(inputTensor.values, h, w, c, iy, ix, ic);
                const weight = kernel3d[ic][ky][kx];
                acc += inputValue * weight;
              }
            }
          }
          const activated = SimEngine.activateValue(acc, p.activation);
          SimEngine.tensor3dSet(output, outH, outW, outC, oy, ox, oc, activated);
        }
      }
    }

    return {
      tensor: {
        kind: 'tensor3d',
        shape: [outH, outW, outC],
        values: output,
        colorMode: outC === 1 ? 'grayscale' : undefined
      },
      transitionNote: `conv2d: k=${k}, stride=${stride}, padding=${pad}, dilation=${dilation}`,
      paramsSummary: [
        `outChannels=${outC}`,
        `kernelSize=${k}`,
        `stride=${stride}`,
        `padding=${pad}`,
        `activation=${p.activation}`
      ]
    };
  }

  private static runPool2DOperator(layer: Pool2DLayer, inputTensor: ForwardTensor): OperatorResult {
    if (inputTensor.shape.length !== 3) {
      throw new Error('Pool2D expects [H, W, C] tensor.');
    }
    const [h, w, c] = inputTensor.shape;
    const k = Math.max(1, layer.params.kernelSize);
    const stride = Math.max(1, layer.params.stride);
    const pad = Math.max(0, layer.params.padding);
    const outH = Math.floor((h + pad * 2 - k) / stride) + 1;
    const outW = Math.floor((w + pad * 2 - k) / stride) + 1;
    if (outH <= 0 || outW <= 0) {
      throw new Error('Pool2D output shape is invalid.');
    }

    const output = new Array(outH * outW * c).fill(0);
    for (let oy = 0; oy < outH; oy += 1) {
      for (let ox = 0; ox < outW; ox += 1) {
        for (let ch = 0; ch < c; ch += 1) {
          let maxVal = -Infinity;
          let sum = 0;
          let count = 0;
          for (let ky = 0; ky < k; ky += 1) {
            for (let kx = 0; kx < k; kx += 1) {
              const iy = oy * stride + ky - pad;
              const ix = ox * stride + kx - pad;
              const val = iy < 0 || iy >= h || ix < 0 || ix >= w
                ? 0
                : SimEngine.tensor3dGet(inputTensor.values, h, w, c, iy, ix, ch);
              maxVal = Math.max(maxVal, val);
              sum += val;
              count += 1;
            }
          }
          const pooled = layer.params.mode === 'avg' ? sum / Math.max(1, count) : maxVal;
          SimEngine.tensor3dSet(output, outH, outW, c, oy, ox, ch, pooled);
        }
      }
    }

    return {
      tensor: {
        kind: 'tensor3d',
        shape: [outH, outW, c],
        values: output,
        colorMode: inputTensor.colorMode
      },
      transitionNote: `pool2d(${layer.params.mode}): k=${k}, stride=${stride}, padding=${pad}`,
      paramsSummary: [
        `mode=${layer.params.mode}`,
        `kernelSize=${k}`,
        `stride=${stride}`,
        `padding=${pad}`
      ]
    };
  }

  private static runFlattenOperator(inputTensor: ForwardTensor): OperatorResult {
    return {
      tensor: {
        kind: 'vector',
        shape: [inputTensor.values.length],
        values: inputTensor.values.slice()
      },
      transitionNote: 'flatten: tensor reshaped into a vector.',
      paramsSummary: ['explicit flatten layer']
    };
  }

  private static runDenseOperator(layer: DenseLayer, inputTensor: ForwardTensor): OperatorResult {
    const inputVector = inputTensor.values;
    const inDim = inputVector.length;
    const units = Math.max(1, layer.params.units);
    const out = new Array(units).fill(0);

    for (let o = 0; o < units; o += 1) {
      let acc = layer.params.bias?.[o] ?? 0;
      for (let i = 0; i < inDim; i += 1) {
        const w = layer.params.weights?.[o]?.[i] ?? SimEngine.syntheticWeight(layer.id, o, i);
        acc += inputVector[i] * w;
      }
      out[o] = SimEngine.activateValue(acc, layer.params.activation);
    }

    return {
      tensor: {
        kind: 'vector',
        shape: [units],
        values: out
      },
      transitionNote: `dense: ${inDim} -> ${units}`,
      paramsSummary: [
        `units=${units}`,
        `activation=${layer.params.activation}`,
        layer.params.weights ? 'weights=custom' : 'weights=generated'
      ]
    };
  }

  private static runActivationOperator(layer: Extract<NetworkLayer, { type: 'activation' }>, inputTensor: ForwardTensor): OperatorResult {
    const activation = layer.params.activationType;
    const nextValues = activation === 'softmax' && inputTensor.shape.length === 1
      ? SimEngine.softmax(inputTensor.values)
      : inputTensor.values.map((value) => SimEngine.activateValue(value, activation));

    return {
      tensor: {
        ...inputTensor,
        values: nextValues
      },
      transitionNote: `activation: ${activation}`,
      paramsSummary: [`activationType=${activation}`]
    };
  }

  private static runDropoutOperator(layer: Extract<NetworkLayer, { type: 'dropout' }>, inputTensor: ForwardTensor): OperatorResult {
    const rate = Math.max(0, Math.min(0.95, layer.params.rate));
    const training = layer.params.training ?? false;
    if (!training) {
      return {
        tensor: { ...inputTensor, values: inputTensor.values.slice() },
        transitionNote: 'dropout skipped in inference mode (training=false).',
        paramsSummary: [`rate=${rate}`, 'training=false']
      };
    }

    const keep = 1 - rate;
    const values = inputTensor.values.map((value, idx) => (((idx + 7) % 5 === 0 ? 0 : value) / Math.max(keep, 1e-6)));
    return {
      tensor: { ...inputTensor, values },
      transitionNote: 'dropout applied in training mode.',
      paramsSummary: [`rate=${rate}`, 'training=true']
    };
  }

  private static runOutputOperator(layer: OutputLayer, inputTensor: ForwardTensor): OperatorResult {
    const denseOut = SimEngine.runDenseOperator(
      {
        ...layer,
        type: 'dense',
        params: {
          units: layer.params.units,
          weights: layer.params.weights,
          bias: layer.params.bias,
          activation: layer.params.activation
        }
      },
      inputTensor
    );

    const tensor: ForwardTensor = {
      ...denseOut.tensor,
      labels: layer.params.labels
    };
    return {
      tensor,
      transitionNote: `output: ${SimEngine.formatShapeLabel(inputTensor.shape)} -> ${SimEngine.formatShapeLabel(tensor.shape)}`,
      paramsSummary: [
        `units=${layer.params.units}`,
        `activation=${layer.params.activation}`,
        layer.params.labels ? `labels=${layer.params.labels.length}` : 'labels=none'
      ]
    };
  }

  private static buildLayerVisualization(tensor: ForwardTensor): ForwardLayerResult['visualization'] {
    if (tensor.shape.length === 3) {
      const sampled = SimEngine.downsampleTensor3d(tensor, SimEngine.maxVisualizationSide);
      const [h, w, c] = sampled.shape as [number, number, number];
      const channelPreviews = Array.from({ length: Math.min(c, 4) }, (_, channel) => {
        const values = SimEngine.extractChannel(sampled.values, h, w, c, channel);
        return {
          channel,
          width: w,
          height: h,
          values: SimEngine.normalizeValues(values)
        };
      });

      const main = c === 1
        ? channelPreviews[0]?.values ?? []
        : c === 3 && (sampled.colorMode === 'rgb' || sampled.colorMode === undefined)
          ? SimEngine.normalizeValues(sampled.values)
          : channelPreviews[0]?.values ?? [];

      return {
        mode: 'image',
        width: w,
        height: h,
        channels: c,
        values: main,
        channelPreviews
      };
    }

    if (tensor.shape.length >= 1) {
      return {
        mode: 'vector',
        values: tensor.values.slice(0, 512)
      };
    }

    return {
      mode: 'none',
      values: []
    };
  }

  private static computeTensorStats(tensor: ForwardTensor): TensorStats {
    if (tensor.values.length === 0) {
      return {
        min: 0,
        max: 0,
        mean: 0,
        nonZeroRatio: 0,
        topK: []
      };
    }

    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    let sum = 0;
    let nonZeroCount = 0;
    for (const value of tensor.values) {
      if (value < min) min = value;
      if (value > max) max = value;
      sum += value;
      if (Math.abs(value) > 1e-8) nonZeroCount += 1;
    }
    const mean = sum / tensor.values.length;
    const nonZeroRatio = nonZeroCount / tensor.values.length;

    const indexed = tensor.values.map((value, index) => ({ index, value, label: tensor.labels?.[index] }));
    indexed.sort((a, b) => b.value - a.value);

    return {
      min,
      max,
      mean,
      nonZeroRatio,
      topK: indexed.slice(0, 8)
    };
  }

  private static mockInputFromLayers(layers: NetworkLayer[]): ForwardInputAsset | null {
    const inputLayer = layers.find((layer) => layer.type === 'input');
    if (!inputLayer || inputLayer.type !== 'input') {
      return null;
    }
    const w = Math.max(1, inputLayer.params.width);
    const h = Math.max(1, inputLayer.params.height);
    const c = Math.max(1, inputLayer.params.channels);
    const values = new Array(w * h * c).fill(0.5);
    const original: ForwardTensor = {
      kind: 'tensor3d',
      shape: [h, w, c],
      values,
      colorMode: c === 1 ? 'grayscale' : 'rgb'
    };
    return {
      id: 'mock',
      source: 'dataset',
      name: 'Mock',
      originalWidth: w,
      originalHeight: h,
      originalChannels: c,
      originalColorMode: c === 1 ? 'grayscale' : 'rgb',
      originalTensor: original,
      prepared: {
        tensor: original,
        displayTensor: original,
        notes: []
      }
    };
  }

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

  private static tensor3dGet(values: number[], _h: number, w: number, c: number, y: number, x: number, ch: number): number {
    const idx = ((y * w) + x) * c + ch;
    return values[idx];
  }

  private static tensor3dSet(values: number[], _h: number, w: number, c: number, y: number, x: number, ch: number, value: number): void {
    const idx = ((y * w) + x) * c + ch;
    values[idx] = value;
  }

  private static resolveKernel3d(layer: Conv2DLayer, outChannel: number, inChannels: number, kernelSize: number): number[][][] {
    const kernel = layer.params.kernels?.[outChannel]?.weights;
    if (kernel && kernel.length > 0) {
      return Array.from({ length: inChannels }, (_, inChannel) => {
        const matrix = kernel[inChannel] ?? kernel[kernel.length - 1] ?? layer.params.kernelMatrix ?? SimEngine.edgeKernel3x3;
        return SimEngine.fitKernelMatrix(matrix, kernelSize);
      });
    }

    const single = SimEngine.fitKernelMatrix(layer.params.kernelMatrix ?? SimEngine.edgeKernel3x3, kernelSize);
    return Array.from({ length: inChannels }, () => single.map((row) => [...row]));
  }

  private static fitKernelMatrix(matrix: number[][], kernelSize: number): number[][] {
    const source = matrix.length > 0 ? matrix : SimEngine.edgeKernel3x3;
    return Array.from({ length: kernelSize }, (_, y) =>
      Array.from({ length: kernelSize }, (_, x) => source[y]?.[x] ?? 0)
    );
  }

  private static syntheticWeight(layerSeed: number, outIndex: number, inIndex: number): number {
    return Math.sin((layerSeed + 1) * 0.173 + (outIndex + 1) * 0.119 + (inIndex + 1) * 0.071) * 0.5;
  }

  private static activateValue(value: number, activation: ActivationType): number {
    if (activation === 'none' || activation === 'softmax') {
      return value;
    }
    if (activation === 'relu') {
      return Math.max(0, value);
    }
    if (activation === 'tanh') {
      return Math.tanh(value);
    }
    if (activation === 'gelu') {
      const cdf = 0.5 * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3))));
      return value * cdf;
    }
    return 1 / (1 + Math.exp(-value));
  }

  private static softmax(values: number[]): number[] {
    if (values.length === 0) {
      return [];
    }
    let max = Number.NEGATIVE_INFINITY;
    for (const value of values) {
      if (value > max) max = value;
    }
    const exps = values.map((value) => Math.exp(value - max));
    const sum = exps.reduce((acc, value) => acc + value, 0);
    if (sum <= 0) {
      return values.map(() => 0);
    }
    return exps.map((value) => value / sum);
  }

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

  private static downsampleTensor3d(tensor: ForwardTensor, maxSide: number): ForwardTensor {
    if (tensor.shape.length !== 3) {
      return tensor;
    }
    const [h, w, c] = tensor.shape as [number, number, number];
    if (Math.max(h, w) <= maxSide) {
      return tensor;
    }
    const scale = maxSide / Math.max(h, w);
    const outH = Math.max(1, Math.round(h * scale));
    const outW = Math.max(1, Math.round(w * scale));
    const out = new Array(outH * outW * c);

    for (let y = 0; y < outH; y += 1) {
      const srcY = Math.min(h - 1, Math.floor((y / outH) * h));
      for (let x = 0; x < outW; x += 1) {
        const srcX = Math.min(w - 1, Math.floor((x / outW) * w));
        for (let ch = 0; ch < c; ch += 1) {
          const srcIdx = ((srcY * w) + srcX) * c + ch;
          const dstIdx = ((y * outW) + x) * c + ch;
          out[dstIdx] = tensor.values[srcIdx] ?? 0;
        }
      }
    }

    return {
      ...tensor,
      shape: [outH, outW, c],
      values: out
    };
  }

  private static extractChannel(values: number[], h: number, w: number, c: number, channel: number): number[] {
    const out = new Array(h * w);
    for (let y = 0; y < h; y += 1) {
      for (let x = 0; x < w; x += 1) {
        const source = ((y * w) + x) * c + channel;
        out[y * w + x] = values[source];
      }
    }
    return out;
  }

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

  private static cloneTensor(tensor: ForwardTensor): ForwardTensor {
    return {
      ...tensor,
      shape: [...tensor.shape] as TensorShape,
      values: tensor.values.slice(),
      labels: tensor.labels ? [...tensor.labels] : undefined
    };
  }

  private static shapeElementCount(shape: TensorShape): number {
    if (shape.length === 0) {
      return 0;
    }
    return shape.reduce((acc, value) => acc * value, 1);
  }

  private static kindFromShape(shape: TensorShape): ForwardTensor['kind'] {
    if (shape.length === 0) {
      return 'scalar';
    }
    if (shape.length === 1) {
      return 'vector';
    }
    if (shape.length === 2) {
      return 'matrix';
    }
    return 'tensor3d';
  }

  private static layerUnits(layer: NetworkLayer): number {
    if (layer.type === 'input') {
      return layer.params.width * layer.params.height * layer.params.channels;
    }
    if (layer.type === 'conv2d') {
      return layer.params.outChannels;
    }
    if (layer.type === 'pool2d') {
      return layer.params.kernelSize;
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
