import { Injectable } from '@angular/core';
import {
  ModeCClassScore,
  ModeCConvChannelExample,
  ModeCConvInputContribution,
  ModeCLayerChannelPreview,
  ModeCLayerDetailRuntime,
  ModeCLayerActivationSummary,
  ModeCLayerPreview,
  ModeCPoolChannelExample,
  ModeCReluChannelExample,
  ModeCSamplePrediction
} from '../models/mode-c.types';
import { ModeCAssetsService, RawNetworkLayer } from './mode-c-assets.service';

type Tensor3D = number[][][];
type Tensor1D = number[];
type ActivationValue = Tensor3D | Tensor1D;

declare const tf: any;

@Injectable({ providedIn: 'root' })
export class ModeCInferenceService {
  private readonly classLabels = [
    'lifeboat',
    'ladybug',
    'pizza',
    'bell pepper',
    'bus',
    'koala',
    'espresso',
    'red panda',
    'orange',
    'sport car'
  ];

  private tfReadyPromise: Promise<any> | null = null;
  private modelPromise: Promise<any> | null = null;
  private rawLayersPromise: Promise<RawNetworkLayer[]> | null = null;
  private readonly objectUrls = new Set<string>();

  constructor(private readonly assets: ModeCAssetsService) {}

  async runSample(assetPath: string): Promise<ModeCInferenceResult> {
    const [model, rawLayers, image] = await Promise.all([
      this.getModel(),
      this.getRawLayers(),
      this.loadAndPrepareImage(assetPath)
    ]);

    const inputTensor = tf.tensor3d(image);
    let currentTensor = tf.stack([inputTensor]);
    let current: ActivationValue = image;
    const summaries: ModeCLayerActivationSummary[] = [
      this.summarizeActivation('input', current)
    ];
    const previews: ModeCLayerPreview[] = [
      {
        layerId: 'input',
        dataUrl: this.createPreviewDataUrl('input', current)
      }
    ];
    const details: ModeCLayerDetailRuntime[] = [
      this.createLayerDetail('input', current)
    ];

    for (let index = 0; index < model.layers.length; index++) {
      const layer = model.layers[index];
      const rawLayer = rawLayers[index];
      const previousActivation = current;
      const nextTensor = layer.apply(currentTensor);
      const squeezedTensor = nextTensor.squeeze();
      current = await squeezedTensor.array();

      summaries.push(this.summarizeActivation(layer.name, current));
      previews.push({
        layerId: layer.name,
        dataUrl: this.createPreviewDataUrl(layer.name, current)
      });
      details.push(this.createLayerDetail(layer.name, current, rawLayer, previousActivation));
      currentTensor = nextTensor;
    }

    const probabilities = this.isTensor3D(current) ? [] : current;
    const topClasses = this.buildTopClasses(probabilities);
    const topClass = topClasses[0] ?? { label: 'unknown', score: 0 };

    return {
      prediction: {
        label: topClass.label,
        confidence: topClass.score,
        topClasses
      },
      summaries,
      previews,
      details
    };
  }

  private async getTf(): Promise<any> {
    const existingTf = typeof window !== 'undefined' ? (window as Window & { tf?: any }).tf : undefined;
    if (existingTf) {
      return existingTf;
    }

    if (!this.tfReadyPromise) {
      this.tfReadyPromise = new Promise((resolve, reject) => {
        const existingScript = document.querySelector('script[data-mode-c-tf="true"]') as HTMLScriptElement | null;
        if (existingScript) {
          existingScript.addEventListener('load', () => resolve((window as Window & { tf?: any }).tf));
          existingScript.addEventListener('error', () => reject(new Error('Failed to load TensorFlow.js for Mode C.')));
          return;
        }

        const script = document.createElement('script');
        script.src = '/modules/cnn-explainer/vendor/tf.min.js';
        script.async = true;
        script.dataset['modeCTf'] = 'true';
        script.onload = () => resolve((window as Window & { tf?: any }).tf);
        script.onerror = () => reject(new Error('Failed to load TensorFlow.js for Mode C.'));
        document.head.appendChild(script);
      });
    }

    return this.tfReadyPromise;
  }

  private async getModel(): Promise<any> {
    if (!this.modelPromise) {
      this.modelPromise = this.getTf().then(tfLib => tfLib.loadLayersModel(this.assets.modelDataUrl));
    }

    return this.modelPromise;
  }

  private async getRawLayers(): Promise<RawNetworkLayer[]> {
    if (!this.rawLayersPromise) {
      this.rawLayersPromise = this.buildRawLayersFromModel();
    }
    return this.rawLayersPromise;
  }

  private async buildRawLayersFromModel(): Promise<RawNetworkLayer[]> {
    const model = await this.getModel();
    const rawLayers: RawNetworkLayer[] = [];

    for (const layer of model.layers) {
      const inputShape = Array.isArray(layer.batchInputShape)
        ? layer.batchInputShape.slice(1)
        : Array.isArray(layer.input?.shape)
          ? layer.input.shape.slice(1)
          : [];
      const outputShape = Array.isArray(layer.outputShape)
        ? layer.outputShape.slice(1)
        : [];
      const layerWeights = layer.getWeights?.() ?? [];

      if (layer.name.includes('conv') && layerWeights.length >= 2) {
        const kernel = await layerWeights[0].array();
        const biases = await layerWeights[1].array();
        const kernelHeight = kernel.length;
        const kernelWidth = kernel[0]?.length ?? 0;
        const inputChannels = kernel[0]?.[0]?.length ?? 0;
        const outputChannels = kernel[0]?.[0]?.[0]?.length ?? 0;
        const weights = Array.from({ length: outputChannels }, (_, outputChannelIndex) => ({
          bias: biases[outputChannelIndex] ?? 0,
          weights: Array.from({ length: kernelHeight }, (_, kernelRow) =>
            Array.from({ length: kernelWidth }, (_, kernelCol) =>
              Array.from({ length: inputChannels }, (_, inputChannelIndex) =>
                kernel[kernelRow]?.[kernelCol]?.[inputChannelIndex]?.[outputChannelIndex] ?? 0
              )
            )
          )
        }));

        rawLayers.push({
          name: layer.name,
          input_shape: inputShape,
          output_shape: outputShape,
          num_neurons: outputChannels,
          weights
        });
        continue;
      }

      if (layer.name.includes('output') && layerWeights.length >= 2) {
        const kernel = await layerWeights[0].array();
        const biases = await layerWeights[1].array();
        const outputUnits = biases.length ?? 0;
        const inputUnits = kernel.length ?? 0;
        const weights = Array.from({ length: outputUnits }, (_, outputIndex) => ({
          bias: biases[outputIndex] ?? 0,
          weights: Array.from({ length: inputUnits }, (_, inputIndex) => kernel[inputIndex]?.[outputIndex] ?? 0)
        }));

        rawLayers.push({
          name: layer.name,
          input_shape: inputShape,
          output_shape: outputShape,
          num_neurons: outputUnits,
          weights
        });
        continue;
      }

      rawLayers.push({
        name: layer.name,
        input_shape: inputShape,
        output_shape: outputShape,
        num_neurons: outputShape[2] ?? outputShape[0] ?? 0,
        weights: []
      });
    }

    return rawLayers;
  }

  private async loadAndPrepareImage(assetPath: string): Promise<number[][][]> {
    const image = await this.loadImage(assetPath);
    const size = 64;
    const canvas = document.createElement('canvas');
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext('2d');
    if (!context) {
      throw new Error('Failed to create 2D canvas context for Mode C inference.');
    }

    let imageData: ImageData;

    if (image.naturalWidth > size || image.naturalHeight > size) {
      const resizeCanvas = document.createElement('canvas');
      const resizeContext = resizeCanvas.getContext('2d');
      if (!resizeContext) {
        throw new Error('Failed to create resize canvas context for Mode C inference.');
      }

      const smallerDimension = Math.min(image.naturalWidth, image.naturalHeight);
      const resizeFactor = (size + 1) / smallerDimension;
      resizeCanvas.width = Math.round(image.naturalWidth * resizeFactor);
      resizeCanvas.height = Math.round(image.naturalHeight * resizeFactor);
      resizeContext.drawImage(image, 0, 0, resizeCanvas.width, resizeCanvas.height);

      if (image.naturalWidth !== image.naturalHeight) {
        context.translate(resizeCanvas.width, 0);
        context.scale(-1, 1);
        context.translate(resizeCanvas.width / 2, resizeCanvas.height / 2);
        context.rotate((90 * Math.PI) / 180);
        context.drawImage(resizeCanvas, -resizeCanvas.width / 2, -resizeCanvas.height / 2);
      } else {
        context.drawImage(resizeCanvas, 0, 0);
      }

      imageData = context.getImageData(0, 0, resizeCanvas.width, resizeCanvas.height);
    } else {
      context.drawImage(image, 0, 0);
      imageData = context.getImageData(0, 0, image.naturalWidth, image.naturalHeight);
    }

    return this.imageDataToTensor(imageData.data, imageData.width, imageData.height);
  }

  private loadImage(assetPath: string): Promise<HTMLImageElement> {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.crossOrigin = 'Anonymous';
      image.onload = () => resolve(image);
      image.onerror = () => reject(new Error(`Failed to load sample image: ${assetPath}`));
      image.src = assetPath;
    });
  }

  private imageDataToTensor(imageData: Uint8ClampedArray, width: number, height: number): Tensor3D {
    let output = Array.from({ length: width }, () =>
      Array.from({ length: height }, () => Array.from({ length: 3 }, () => 0))
    );

    for (let index = 0; index < imageData.length; index++) {
      const pixelIndex = Math.floor(index / 4);
      const channelIndex = index % 4;
      const row = width === height
        ? Math.floor(pixelIndex / width)
        : pixelIndex % width;
      const column = width === height
        ? pixelIndex % width
        : Math.floor(pixelIndex / width);

      if (channelIndex < 3) {
        output[row][column][channelIndex] = imageData[index] / 255;
      }
    }

    if (width !== 64 && height !== 64) {
      output = this.cropCentralSquare(output);
    }

    return output;
  }

  private cropCentralSquare(input: Tensor3D): Tensor3D {
    const width = input.length;
    const height = input[0]?.length ?? 0;
    const cropSize = Math.min(width, height, 64);
    const startX = Math.floor(width / 2) - Math.floor(cropSize / 2);
    const startY = Math.floor(height / 2) - Math.floor(cropSize / 2);
    const cropped = input
      .slice(startX, startX + cropSize)
      .map(row => row.slice(startY, startY + cropSize));

    if (cropSize === 64) {
      return cropped;
    }

    const padded = Array.from({ length: 64 }, () =>
      Array.from({ length: 64 }, () => Array.from({ length: 3 }, () => 0))
    );
    const offset = Math.floor((64 - cropSize) / 2);

    for (let row = 0; row < cropSize; row++) {
      for (let col = 0; col < cropSize; col++) {
        padded[row + offset][col + offset] = cropped[row]?.[col] ?? [0, 0, 0];
      }
    }

    return padded;
  }

  private conv2d(input: number[][][], layer: RawNetworkLayer): number[][][] {
    const outputHeight = layer.output_shape[0];
    const outputWidth = layer.output_shape[1];
    const outputChannels = layer.output_shape[2];
    const kernelSize = Array.isArray(layer.weights[0]?.weights) ? layer.weights[0].weights.length : 0;
    const inputChannels = input[0]?.[0]?.length ?? 0;
    const output = Array.from({ length: outputHeight }, () =>
      Array.from({ length: outputWidth }, () => Array.from({ length: outputChannels }, () => 0))
    );

    for (let outChannel = 0; outChannel < outputChannels; outChannel++) {
      const filter = layer.weights[outChannel];
      for (let row = 0; row < outputHeight; row++) {
        for (let col = 0; col < outputWidth; col++) {
          let value = filter.bias;
          for (let kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
            for (let kernelCol = 0; kernelCol < kernelSize; kernelCol++) {
              for (let inChannel = 0; inChannel < inputChannels; inChannel++) {
                const kernelPlane = filter.weights?.[kernelRow] as unknown[] | undefined;
                const kernelRowValues = Array.isArray(kernelPlane?.[kernelCol])
                  ? kernelPlane[kernelCol] as number[]
                  : [];
                value += (input[row + kernelRow]?.[col + kernelCol]?.[inChannel] ?? 0) *
                  (kernelRowValues[inChannel] ?? 0);
              }
            }
          }
          output[row][col][outChannel] = value;
        }
      }
    }

    return output;
  }

  private relu(input: number[][][]): number[][][] {
    return input.map(row =>
      row.map(pixel =>
        pixel.map(value => Math.max(0, value))
      )
    );
  }

  private maxPool2x2(input: number[][][]): number[][][] {
    const inputHeight = input.length;
    const inputWidth = input[0]?.length ?? 0;
    const channels = input[0]?.[0]?.length ?? 0;
    const outputHeight = Math.floor(inputHeight / 2);
    const outputWidth = Math.floor(inputWidth / 2);
    const output = Array.from({ length: outputHeight }, () =>
      Array.from({ length: outputWidth }, () => Array.from({ length: channels }, () => 0))
    );

    for (let row = 0; row < outputHeight; row++) {
      for (let col = 0; col < outputWidth; col++) {
        for (let channel = 0; channel < channels; channel++) {
          let max = -Infinity;
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              max = Math.max(max, input[row * 2 + dy]?.[col * 2 + dx]?.[channel] ?? -Infinity);
            }
          }
          output[row][col][channel] = max;
        }
      }
    }

    return output;
  }

  private flatten(input: number[][][]): number[] {
    const output: number[] = [];
    const height = input.length;
    const width = input[0]?.length ?? 0;
    const channels = input[0]?.[0]?.length ?? 0;

    // Match the original tfjs model path: channel -> row -> column.
    for (let channel = 0; channel < channels; channel++) {
      for (let row = 0; row < height; row++) {
        for (let col = 0; col < width; col++) {
          output.push(input[row]?.[col]?.[channel] ?? 0);
        }
      }
    }

    return output;
  }

  private dense(input: number[], layer: RawNetworkLayer): number[] {
    return layer.weights.map(neuron => {
      let value = neuron.bias;
      for (let index = 0; index < input.length; index++) {
        value += input[index] * ((neuron.weights?.[index] as number | undefined) ?? 0);
      }
      return value;
    });
  }

  private summarizeActivation(layerId: string, activation: number[][][] | number[]): ModeCLayerActivationSummary {
    const values = this.flattenValues(activation);
    const count = values.length || 1;
    let min = Number.POSITIVE_INFINITY;
    let max = Number.NEGATIVE_INFINITY;
    let sum = 0;
    let positiveCount = 0;
    let energy = 0;

    for (const value of values) {
      min = Math.min(min, value);
      max = Math.max(max, value);
      sum += value;
      if (value > 0) positiveCount++;
      energy += Math.abs(value);
    }

    return {
      layerId,
      min: Number.isFinite(min) ? min : 0,
      max: Number.isFinite(max) ? max : 0,
      mean: sum / count,
      positiveRatio: positiveCount / count,
      energy: energy / count
    };
  }

  private flattenValues(value: number[][][] | number[]): number[] {
    if (!Array.isArray(value[0])) {
      return value as number[];
    }

    const output: number[] = [];
    const input = value as number[][][];
    for (const row of input) {
      for (const pixel of row) {
        for (const channel of pixel) {
          output.push(channel);
        }
      }
    }
    return output;
  }

  private softmax(logits: number[]): number[] {
    const maxLogit = Math.max(...logits);
    const exps = logits.map(value => Math.exp(value - maxLogit));
    const sum = exps.reduce((acc, value) => acc + value, 0);
    return exps.map(value => value / sum);
  }

  private buildTopClasses(probabilities: number[]): ModeCClassScore[] {
    return probabilities
      .map((score, index) => ({
        label: this.classLabels[index] ?? `class-${index}`,
        score
      }))
      .sort((a, b) => b.score - a.score);
  }

  private createLayerDetail(
    layerId: string,
    activation: ActivationValue,
    rawLayer?: RawNetworkLayer,
    previousActivation?: ActivationValue
  ): ModeCLayerDetailRuntime {
    if (this.isTensor3D(activation)) {
      const channelPreviews = this.createChannelPreviews(activation, layerId === 'input');
      const convExamples = rawLayer && previousActivation && this.isTensor3D(previousActivation) && layerId.includes('conv')
        ? this.safeBuild(() => this.createConvExamples(rawLayer, previousActivation, activation), [])
        : [];
      const reluExamples = previousActivation && this.isTensor3D(previousActivation) && layerId.includes('relu')
        ? this.safeBuild(() => this.createReluExamples(previousActivation, activation), [])
        : [];
      const poolExamples = previousActivation && this.isTensor3D(previousActivation) && layerId.includes('pool')
        ? this.safeBuild(() => this.createPoolExamples(previousActivation, activation), [])
        : [];

      return {
        layerId,
        channelPreviews,
        vectorValues: [],
        convExamples,
        reluExamples,
        poolExamples
      };
    }

    return {
      layerId,
      channelPreviews: [],
      vectorValues: activation.slice(0, 24),
      convExamples: [],
      reluExamples: [],
      poolExamples: []
    };
  }

  private createConvExamples(
    rawLayer: RawNetworkLayer,
    inputActivation: Tensor3D,
    outputActivation: Tensor3D
  ): ModeCConvChannelExample[] {
    const outputChannels = Math.min(outputActivation[0]?.[0]?.length ?? 0, 10);
    const examples: ModeCConvChannelExample[] = [];

    for (let outputChannelIndex = 0; outputChannelIndex < outputChannels; outputChannelIndex++) {
      const focusPoint = this.findMaxActivationPosition(outputActivation, outputChannelIndex);
      const filter = rawLayer.weights[outputChannelIndex];
      const kernelSize = Array.isArray(filter?.weights) ? filter.weights.length : 0;
      const contributions: Array<{ inputChannelIndex: number; score: number }> = [];

      for (let inputChannelIndex = 0; inputChannelIndex < (inputActivation[0]?.[0]?.length ?? 0); inputChannelIndex++) {
        let score = 0;
        for (let kernelRow = 0; kernelRow < kernelSize; kernelRow++) {
          for (let kernelCol = 0; kernelCol < kernelSize; kernelCol++) {
            const weight = this.readKernelWeight(filter, kernelRow, kernelCol, inputChannelIndex);
            const value = inputActivation[focusPoint.row + kernelRow]?.[focusPoint.col + kernelCol]?.[inputChannelIndex] ?? 0;
            score += Math.abs(weight * value);
          }
        }
        contributions.push({ inputChannelIndex, score });
      }

      contributions.sort((a, b) => b.score - a.score);
      const selectedInputChannel = contributions[0]?.inputChannelIndex ?? 0;
      const topContributions = contributions.slice(0, Math.min(4, contributions.length));

      const patch = this.extractPatch(inputActivation, focusPoint.row, focusPoint.col, selectedInputChannel, kernelSize);
      const kernel = this.extractKernel(filter, selectedInputChannel, kernelSize);
      const products = patch.map((patchRow, rowIndex) =>
        patchRow.map((value, colIndex) => value * kernel[rowIndex][colIndex])
      );
      const inputContributionDetails = topContributions.map<ModeCConvInputContribution>(({ inputChannelIndex, score }) => {
        const channelPatch = this.extractPatch(inputActivation, focusPoint.row, focusPoint.col, inputChannelIndex, kernelSize);
        const channelKernel = this.extractKernel(filter, inputChannelIndex, kernelSize);
        const channelProducts = channelPatch.map((patchRow, rowIndex) =>
          patchRow.map((value, colIndex) => value * channelKernel[rowIndex][colIndex])
        );
        const channelWeightedSum = channelProducts.flat().reduce((sum, value) => sum + value, 0);

        return {
          inputChannelIndex,
          contributionScore: score,
          weightedSum: channelWeightedSum,
          patch: channelPatch,
          kernel: channelKernel,
          products: channelProducts,
          patchPreviewUrl: this.createScalarPreviewDataUrl(channelPatch),
          kernelPreviewUrl: this.createScalarPreviewDataUrl(channelKernel)
        };
      });
      const weightedSum = products.flat().reduce((sum, value) => sum + value, 0);
      const outputValue = outputActivation[focusPoint.row][focusPoint.col][outputChannelIndex];

      examples.push({
        outputChannelIndex,
        inputChannelIndex: selectedInputChannel,
        row: focusPoint.row,
        col: focusPoint.col,
        patch,
        kernel,
        products,
        bias: filter?.bias ?? 0,
        weightedSum,
        outputValue,
        patchPreviewUrl: this.createScalarPreviewDataUrl(patch),
        kernelPreviewUrl: this.createScalarPreviewDataUrl(kernel),
        inputContributions: inputContributionDetails
      });
    }

    return examples;
  }

  private createChannelPreviews(input: Tensor3D, grayscale = false): ModeCLayerChannelPreview[] {
    const channelCount = input[0]?.[0]?.length ?? 0;
    const previews: ModeCLayerChannelPreview[] = [];

    for (let channel = 0; channel < Math.min(channelCount, 10); channel++) {
      const matrix = input.map(row => row.map(pixel => pixel[channel] ?? 0));
      const values = matrix.flat();
      const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
      const energy = values.reduce((sum, value) => sum + Math.abs(value), 0) / Math.max(1, values.length);
      previews.push({
        index: channel,
        dataUrl: this.createScalarPreviewDataUrl(matrix, { grayscale }),
        matrix,
        grayscale,
        mean,
        energy
      });
    }

    return previews;
  }

  private createReluExamples(before: Tensor3D, after: Tensor3D): ModeCReluChannelExample[] {
    const channelCount = Math.min(before[0]?.[0]?.length ?? 0, 10);
    const examples: ModeCReluChannelExample[] = [];

    for (let channelIndex = 0; channelIndex < channelCount; channelIndex++) {
      const beforeMatrix = before.map(row => row.map(pixel => pixel[channelIndex] ?? 0));
      const afterMatrix = after.map(row => row.map(pixel => pixel[channelIndex] ?? 0));
      const beforeValues = beforeMatrix.flat();
      const afterValues = afterMatrix.flat();
      const beforeNegativeRatio = beforeValues.filter(value => value < 0).length / Math.max(1, beforeValues.length);
      const afterPositiveRatio = afterValues.filter(value => value > 0).length / Math.max(1, afterValues.length);
      examples.push({
        channelIndex,
        beforePreviewUrl: this.createScalarPreviewDataUrl(beforeMatrix),
        afterPreviewUrl: this.createScalarPreviewDataUrl(afterMatrix),
        beforeNegativeRatio,
        afterPositiveRatio,
        beforeMin: Math.min(...beforeValues),
        afterMin: Math.min(...afterValues),
        afterMax: Math.max(...afterValues)
      });
    }

    return examples;
  }

  private createPoolExamples(before: Tensor3D, after: Tensor3D): ModeCPoolChannelExample[] {
    const channelCount = Math.min(after[0]?.[0]?.length ?? 0, 10);
    const examples: ModeCPoolChannelExample[] = [];

    for (let channelIndex = 0; channelIndex < channelCount; channelIndex++) {
      const focusPoint = this.findMaxActivationPosition(after, channelIndex);
      const patch = [
        [
          before[focusPoint.row * 2]?.[focusPoint.col * 2]?.[channelIndex] ?? 0,
          before[focusPoint.row * 2]?.[focusPoint.col * 2 + 1]?.[channelIndex] ?? 0
        ],
        [
          before[focusPoint.row * 2 + 1]?.[focusPoint.col * 2]?.[channelIndex] ?? 0,
          before[focusPoint.row * 2 + 1]?.[focusPoint.col * 2 + 1]?.[channelIndex] ?? 0
        ]
      ];

      examples.push({
        channelIndex,
        row: focusPoint.row,
        col: focusPoint.col,
        patch,
        maxValue: after[focusPoint.row][focusPoint.col][channelIndex],
        patchPreviewUrl: this.createScalarPreviewDataUrl(patch)
      });
    }

    return examples;
  }

  private createPreviewDataUrl(layerId: string, activation: ActivationValue): string {
    const previewSize = 32;
    const canvas = document.createElement('canvas');
    canvas.width = previewSize;
    canvas.height = previewSize;
    const context = canvas.getContext('2d');
    if (!context) {
      return '';
    }

    const imageData = context.createImageData(previewSize, previewSize);
    const pixels = imageData.data;

    if (this.isTensor3D(activation)) {
      const height = activation.length;
      const width = activation[0]?.length ?? 0;
      const channels = activation[0]?.[0]?.length ?? 0;

      if (layerId === 'input' && channels >= 3) {
        for (let row = 0; row < previewSize; row++) {
          for (let col = 0; col < previewSize; col++) {
            const sourceRow = Math.min(height - 1, Math.floor((row / previewSize) * height));
            const sourceCol = Math.min(width - 1, Math.floor((col / previewSize) * width));
            const index = (row * previewSize + col) * 4;
            pixels[index] = Math.round((activation[sourceRow][sourceCol][0] ?? 0) * 255);
            pixels[index + 1] = Math.round((activation[sourceRow][sourceCol][1] ?? 0) * 255);
            pixels[index + 2] = Math.round((activation[sourceRow][sourceCol][2] ?? 0) * 255);
            pixels[index + 3] = 255;
          }
        }
      } else {
        const projection = this.selectRepresentativeChannel(activation);
        this.paintScalarField(projection, pixels, previewSize, previewSize, false);
      }
    } else {
      const side = Math.max(1, Math.ceil(Math.sqrt(activation.length)));
      const matrix = Array.from({ length: side }, (_, row) =>
        Array.from({ length: side }, (_, col) => activation[row * side + col] ?? 0)
      );
      this.paintScalarField(matrix, pixels, previewSize, previewSize, false);
    }

    context.putImageData(imageData, 0, 0);
    return this.createObjectUrlFromCanvas(canvas);
  }

  private createScalarPreviewDataUrl(
    matrix: number[][],
    options: { grayscale?: boolean } = {}
  ): string {
    const previewSize = 36;
    const canvas = document.createElement('canvas');
    canvas.width = previewSize;
    canvas.height = previewSize;
    const context = canvas.getContext('2d');
    if (!context) {
      return '';
    }

    const imageData = context.createImageData(previewSize, previewSize);
    this.paintScalarField(matrix, imageData.data, previewSize, previewSize, options.grayscale ?? false);
    context.putImageData(imageData, 0, 0);
    return this.createObjectUrlFromCanvas(canvas);
  }

  private findMaxActivationPosition(outputActivation: Tensor3D, outputChannelIndex: number): { row: number; col: number } {
    let bestRow = 0;
    let bestCol = 0;
    let bestValue = Number.NEGATIVE_INFINITY;

    for (let row = 0; row < outputActivation.length; row++) {
      for (let col = 0; col < (outputActivation[row]?.length ?? 0); col++) {
        const value = outputActivation[row][col][outputChannelIndex] ?? Number.NEGATIVE_INFINITY;
        if (value > bestValue) {
          bestValue = value;
          bestRow = row;
          bestCol = col;
        }
      }
    }

    return { row: bestRow, col: bestCol };
  }

  private extractPatch(
    inputActivation: Tensor3D,
    row: number,
    col: number,
    inputChannelIndex: number,
    kernelSize: number
  ): number[][] {
    return Array.from({ length: kernelSize }, (_, kernelRow) =>
      Array.from({ length: kernelSize }, (_, kernelCol) =>
        inputActivation[row + kernelRow]?.[col + kernelCol]?.[inputChannelIndex] ?? 0
      )
    );
  }

  private extractKernel(
    filter: RawNetworkLayer['weights'][number],
    inputChannelIndex: number,
    kernelSize: number
  ): number[][] {
    return Array.from({ length: kernelSize }, (_, kernelRow) =>
      Array.from({ length: kernelSize }, (_, kernelCol) =>
        this.readKernelWeight(filter, kernelRow, kernelCol, inputChannelIndex)
      )
    );
  }

  private readKernelWeight(
    filter: RawNetworkLayer['weights'][number],
    kernelRow: number,
    kernelCol: number,
    inputChannelIndex: number
  ): number {
    const plane = filter.weights?.[kernelRow] as unknown[] | undefined;
    const rowValues = Array.isArray(plane?.[kernelCol]) ? plane[kernelCol] as number[] : [];
    return rowValues[inputChannelIndex] ?? 0;
  }

  private selectRepresentativeChannel(input: Tensor3D): number[][] {
    const channels = input[0]?.[0]?.length ?? 0;
    let bestChannel = 0;
    let bestEnergy = Number.NEGATIVE_INFINITY;

    for (let channel = 0; channel < channels; channel++) {
      let energy = 0;
      for (let row = 0; row < input.length; row++) {
        for (let col = 0; col < (input[row]?.length ?? 0); col++) {
          energy += Math.abs(input[row][col][channel] ?? 0);
        }
      }
      if (energy > bestEnergy) {
        bestEnergy = energy;
        bestChannel = channel;
      }
    }

    return input.map(row => row.map(pixel => pixel[bestChannel] ?? 0));
  }

  private paintScalarField(
    matrix: number[][],
    pixels: Uint8ClampedArray,
    targetHeight: number,
    targetWidth: number,
    grayscale: boolean
  ): void {
    const sourceHeight = matrix.length;
    const sourceWidth = matrix[0]?.length ?? 0;
    const flattened = matrix.flat().filter(value => Number.isFinite(value));
    const hasNegative = flattened.some(value => value < 0);
    const hasPositive = flattened.some(value => value > 0);
    const maxAbs = this.computePercentile(flattened.map(value => Math.abs(value)), 0.92);
    const minValue = flattened.length ? Math.min(...flattened) : 0;
    const maxValue = flattened.length ? Math.max(...flattened) : 1;
    const signedRange = maxAbs > 1e-6 ? maxAbs : 1;
    const positiveRange = maxValue > 1e-6 ? maxValue : 1;
    const negativeRange = Math.abs(minValue) > 1e-6 ? Math.abs(minValue) : 1;

    for (let row = 0; row < targetHeight; row++) {
      for (let col = 0; col < targetWidth; col++) {
        const sourceRow = Math.min(sourceHeight - 1, Math.floor((row / targetHeight) * sourceHeight));
        const sourceCol = Math.min(sourceWidth - 1, Math.floor((col / targetWidth) * sourceWidth));
        const value = matrix[sourceRow][sourceCol];
        let color: [number, number, number];

        if (grayscale) {
          color = this.mapGrayscaleColor(Math.max(0, Math.min(1, value)));
        } else if (hasNegative && hasPositive) {
          color = this.mapActivationColor(value / signedRange);
        } else if (hasPositive) {
          color = this.mapPositiveActivationColor(value / positiveRange);
        } else {
          color = this.mapNegativeActivationColor(value / negativeRange);
        }

        const index = (row * targetWidth + col) * 4;
        pixels[index] = color[0];
        pixels[index + 1] = color[1];
        pixels[index + 2] = color[2];
        pixels[index + 3] = 255;
      }
    }
  }

  private mapActivationColor(normalized: number): [number, number, number] {
    const clamped = Math.max(-1, Math.min(1, normalized));
    const intensity = Math.pow(Math.abs(clamped), 0.55);
    const white = 250 - Math.round(intensity * 10);

    if (clamped >= 0) {
      const r = Math.round(white - intensity * 70);
      const g = Math.round(white - intensity * 110);
      const b = Math.round(white + intensity * 4);
      return [r, g, b];
    }

    const r = Math.round(white + intensity * 4);
    const g = Math.round(white - intensity * 108);
    const b = Math.round(white - intensity * 70);
    return [r, g, b];
  }

  private mapGrayscaleColor(normalized: number): [number, number, number] {
    const clamped = Math.max(0, Math.min(1, normalized));
    const value = Math.round(clamped * 255);
    return [value, value, value];
  }

  private mapPositiveActivationColor(normalized: number): [number, number, number] {
    const clamped = Math.max(0, Math.min(1, normalized));
    const intensity = Math.pow(clamped, 0.45);
    return [
      Math.round(248 - intensity * 82),
      Math.round(250 - intensity * 120),
      255
    ];
  }

  private mapNegativeActivationColor(normalized: number): [number, number, number] {
    const clamped = Math.max(-1, Math.min(0, normalized));
    const intensity = Math.pow(Math.abs(clamped), 0.45);
    return [
      255,
      Math.round(246 - intensity * 116),
      Math.round(246 - intensity * 82)
    ];
  }

  private computePercentile(values: number[], percentile: number): number {
    if (!values.length) return 0;
    const sorted = [...values].sort((left, right) => left - right);
    const position = Math.min(sorted.length - 1, Math.max(0, Math.floor(percentile * (sorted.length - 1))));
    return sorted[position] ?? 0;
  }

  private isTensor3D(value: ActivationValue): value is Tensor3D {
    return Array.isArray(value[0]);
  }

  private createObjectUrlFromCanvas(canvas: HTMLCanvasElement): string {
    const dataUrl = canvas.toDataURL('image/png');
    const blob = this.dataUrlToBlob(dataUrl);
    const objectUrl = URL.createObjectURL(blob);
    this.objectUrls.add(objectUrl);
    return objectUrl;
  }

  private dataUrlToBlob(dataUrl: string): Blob {
    const [header, base64] = dataUrl.split(',');
    const mimeMatch = header.match(/data:(.*?);base64/);
    const mimeType = mimeMatch?.[1] ?? 'image/png';
    const binary = atob(base64 ?? '');
    const bytes = new Uint8Array(binary.length);

    for (let index = 0; index < binary.length; index++) {
      bytes[index] = binary.charCodeAt(index);
    }

    return new Blob([bytes], { type: mimeType });
  }

  private safeBuild<T>(factory: () => T, fallback: T): T {
    try {
      return factory();
    } catch {
      return fallback;
    }
  }
}

export interface ModeCInferenceResult {
  prediction: ModeCSamplePrediction;
  summaries: ModeCLayerActivationSummary[];
  previews: ModeCLayerPreview[];
  details: ModeCLayerDetailRuntime[];
}
