import { Injectable } from '@angular/core';
import {
  ModeCArticleSection,
  ModeCDetailTopic,
  ModeCMilestone,
  ModeCNetworkLayer,
  ModeCOverviewStage,
  ModeCSampleOption
} from '../models/mode-c.types';

@Injectable({ providedIn: 'root' })
export class ModeCAssetsService {
  readonly modelDataUrl = '/mode-c/cnn-explainer/assets/data/model.json';

  readonly sampleOptions: ModeCSampleOption[] = [
    {
      id: 'espresso',
      title: '浓缩咖啡',
      label: '咖啡杯样例',
      description: '默认教学样例，适合观察卷积层如何逐步聚焦杯口、杯身和高对比边缘。',
      assetPath: '/mode-c/cnn-explainer/assets/img/espresso_1.jpeg',
      predictedClass: '',
      confidence: 0,
      topClasses: []
    },
    {
      id: 'panda',
      title: '熊猫',
      label: '动物样例',
      description: '黑白对比明显，适合观察卷积通道对毛发纹理和轮廓区域的不同响应。',
      assetPath: '/mode-c/cnn-explainer/assets/img/panda_1.jpeg',
      predictedClass: '',
      confidence: 0,
      topClasses: []
    },
    {
      id: 'pizza',
      title: '披萨',
      label: '食物样例',
      description: '纹理丰富，适合比较不同卷积通道对边缘、块状结构和表面细节的关注差异。',
      assetPath: '/mode-c/cnn-explainer/assets/img/pizza_1.jpeg',
      predictedClass: '',
      confidence: 0,
      topClasses: []
    },
    {
      id: 'bus',
      title: '公交车',
      label: '交通工具样例',
      description: '几何结构清晰，适合观察模型如何在较规整的目标上形成分类判断。',
      assetPath: '/mode-c/cnn-explainer/assets/img/bus_1.jpeg',
      predictedClass: '',
      confidence: 0,
      topClasses: []
    }
  ];

  readonly fallbackNetworkLayers: ModeCNetworkLayer[] = [
    this.buildFallbackLayer('input', '输入图像', '输入', 'input', 3, 64, '64 x 64 x 3', '64 x 64 x 3', null, 0, '输入样本会先被整理成 64×64 的 RGB 张量，再送入 CNN。', 'encoder-a'),
    this.buildFallbackLayer('conv_1_1', '卷积 1.1', '卷积 1.1', 'conv', 10, 62, '64 x 64 x 3', '62 x 62 x 10', '3 x 3', 280, '第一层卷积负责提取边缘、明暗变化和基础纹理。', 'encoder-a'),
    this.buildFallbackLayer('relu_1_1', 'ReLU 1.1', 'ReLU 1.1', 'relu', 10, 62, '62 x 62 x 10', '62 x 62 x 10', null, 0, 'ReLU 会保留正向响应并压制负值，形成更稀疏的特征图。', 'encoder-a'),
    this.buildFallbackLayer('conv_1_2', '卷积 1.2', '卷积 1.2', 'conv', 10, 60, '62 x 62 x 10', '60 x 60 x 10', '3 x 3', 910, '第二个卷积层在上一层基础上继续组合局部模式。', 'encoder-a'),
    this.buildFallbackLayer('relu_1_2', 'ReLU 1.2', 'ReLU 1.2', 'relu', 10, 60, '60 x 60 x 10', '60 x 60 x 10', null, 0, '继续保留更有判别力的响应，为池化前的特征压缩做准备。', 'encoder-a'),
    this.buildFallbackLayer('max_pool_1', '最大池化 1', '池化 1', 'pool', 10, 30, '60 x 60 x 10', '30 x 30 x 10', '2 x 2', 0, '第一次池化降低分辨率，同时尽量保留最强响应。', 'encoder-a'),
    this.buildFallbackLayer('conv_2_1', '卷积 2.1', '卷积 2.1', 'conv', 10, 28, '30 x 30 x 10', '28 x 28 x 10', '3 x 3', 910, '在更紧凑的空间尺度上继续提取更抽象的局部结构。', 'encoder-b'),
    this.buildFallbackLayer('relu_2_1', 'ReLU 2.1', 'ReLU 2.1', 'relu', 10, 28, '28 x 28 x 10', '28 x 28 x 10', null, 0, '保留对分类更有帮助的正向特征。', 'encoder-b'),
    this.buildFallbackLayer('conv_2_2', '卷积 2.2', '卷积 2.2', 'conv', 10, 26, '28 x 28 x 10', '26 x 26 x 10', '3 x 3', 910, '进一步巩固类别相关模式，为最终分类做准备。', 'encoder-b'),
    this.buildFallbackLayer('relu_2_2', 'ReLU 2.2', 'ReLU 2.2', 'relu', 10, 26, '26 x 26 x 10', '26 x 26 x 10', null, 0, '在进入第二次池化前，过滤掉负向响应。', 'encoder-b'),
    this.buildFallbackLayer('max_pool_2', '最大池化 2', '池化 2', 'pool', 10, 13, '26 x 26 x 10', '13 x 13 x 10', '2 x 2', 0, '第二次池化得到更紧凑的高层特征表示。', 'encoder-b'),
    this.buildFallbackLayer('flatten', 'Flatten 层', 'Flatten', 'flatten', 1690, 1, '13 x 13 x 10', '1690', null, 0, '将最后的特征图堆栈展开成一维向量，供输出层使用。', 'bridge'),
    this.buildFallbackLayer('output', '输出层', '输出', 'output', 10, 1, '1690', '10', null, 16910, '对 10 个类别生成 logits，再转成最终概率分布。', 'classifier')
  ];

  readonly overviewStages: ModeCOverviewStage[] = [
    { id: 'input', title: '输入样本', summary: '选择教学样本并准备推理。', status: 'ready' },
    { id: 'graph', title: 'CNN 总览图', summary: '查看各层特征图、通道和输出类别分布。', status: 'ready' },
    { id: 'detail', title: '细节解释', summary: '展开卷积、ReLU、池化和 softmax 的具体过程。', status: 'ready' },
    { id: 'article', title: '教学说明', summary: '补充术语、问题引导和答辩用解释路径。', status: 'ready' }
  ];

  readonly detailTopics: ModeCDetailTopic[] = [
    { id: 'overview-graph', title: '总览图', description: '聚焦当前样本在整张 CNN 图上的层级响应。', priority: 'P0' },
    { id: 'sample-switching', title: '样本切换', description: '比较不同样本在同一模型上的响应差异。', priority: 'P0' },
    { id: 'conv-panel', title: '卷积解释', description: '展示 patch、kernel、products、加权求和与 bias。', priority: 'P1' },
    { id: 'softmax-panel', title: '输出解释', description: '展示类别排序、概率分布与最终预测。', priority: 'P1' }
  ];

  readonly milestones: ModeCMilestone[] = [
    { id: 'shell', title: '原生页面壳', note: 'Mode C 已不再依赖 iframe 宿主。', status: 'ready' },
    { id: 'state', title: '状态服务', note: '样本、图层、通道和推理结果都已纳入 Angular 状态管理。', status: 'ready' },
    { id: 'graph', title: '总览图迁移', note: '主拓扑图和通道特征图已迁移到 Angular。', status: 'ready' },
    { id: 'detail-ready', title: '细节联动', note: '细节面板可随选中层与通道实时切换。', status: 'ready' }
  ];

  readonly articleSections: ModeCArticleSection[] = [
    {
      id: 'goal',
      eyebrow: 'Mode C',
      title: 'CNN 卷积过程与中间特征解释',
      body: [
        'Mode C 聚焦回答“CNN 为什么这样判断”。',
        '它通过静态 TF.js 模型、特征图和中间过程视图，把卷积、激活、池化和输出解释串成一条完整教学链路。'
      ]
    },
    {
      id: 'mapping',
      eyebrow: '解释重点',
      title: '从输入到输出怎么被看懂',
      body: [
        '输入样本会先经过卷积层提取局部模式，再经过 ReLU 和池化逐步压缩成更有判别力的高层特征。',
        '最后 flatten 和输出层把这些特征映射成类别分数，并通过 softmax 形成最终概率排序。'
      ],
      bullets: [
        '看卷积如何从局部 patch 生成单个输出值',
        '看不同通道为何会关注不同结构',
        '看 softmax 如何把特征向量转成分类结果'
      ]
    },
    {
      id: 'next',
      eyebrow: '可解释性',
      title: '当前页面适合怎样的答辩演示',
      body: [
        '先选样本，再看总览图中的高响应通道，随后展开卷积、ReLU、池化和 softmax 的具体过程。',
        '如果需要强调可解释性，可进一步结合通道响应和类别概率讲清模型是依据哪些区域做出判断的。'
      ]
    }
  ];

  async loadNetworkLayers(): Promise<ModeCNetworkLayer[]> {
    const modelConfig = await this.loadModelConfig();
    return this.mapModelLayers(modelConfig);
  }

  async loadModelConfig(): Promise<ModelJsonConfig> {
    const response = await fetch(this.modelDataUrl);
    if (!response.ok) {
      throw new Error(`Failed to load model config: HTTP ${response.status}`);
    }

    return response.json() as Promise<ModelJsonConfig>;
  }

  private mapModelLayers(model: ModelJsonConfig): ModeCNetworkLayer[] {
    const kerasLayers = model.modelTopology.model_config.config.layers;
    const firstConv = kerasLayers.find(layer => Array.isArray(layer.config.batch_input_shape));
    const inputShape = (firstConv?.config.batch_input_shape?.slice(1) as number[] | undefined) ?? [64, 64, 3];
    const layers: ModeCNetworkLayer[] = [
      this.buildFallbackLayer(
        'input',
        '输入图像',
        '输入',
        'input',
        inputShape[2] ?? 3,
        inputShape[0] ?? 64,
        this.formatShape(inputShape),
        this.formatShape(inputShape),
        null,
        0,
        'RGB 输入张量是整个 CNN 推理链路的起点。',
        'encoder-a'
      )
    ];

    let currentShape = [...inputShape];
    for (const layer of kerasLayers) {
      const mapped = this.mapModelLayer(layer, currentShape);
      layers.push(mapped.layer);
      currentShape = mapped.outputShape;
    }

    return layers;
  }

  private mapModelLayer(
    layer: ModelTopologyLayer,
    inputShape: number[]
  ): { layer: ModeCNetworkLayer; outputShape: number[] } {
    const layerType = this.inferLayerType(layer.config.name);
    const outputShape = this.computeModelOutputShape(layer, inputShape, layerType);
    const parameterCount = this.computeModelParameterCount(layer, inputShape, outputShape, layerType);
    const kernelSize = this.inferModelKernelSize(layer, layerType);
    const channels = outputShape.length >= 3 ? outputShape[2] : outputShape[0] ?? 0;
    const spatialSize = outputShape.length >= 2 ? outputShape[0] : 1;

    return {
      layer: {
        id: layer.config.name,
        sourceName: layer.config.name,
        title: this.buildLayerTitle(layer.config.name, layerType),
        shortTitle: this.buildShortTitle(layer.config.name, layerType),
        type: layerType,
        channels,
        spatialSize,
        inputShapeLabel: this.formatShape(inputShape),
        outputShapeLabel: this.formatShape(outputShape),
        kernelLabel: kernelSize ? `${kernelSize} x ${kernelSize}` : null,
        parameterCount,
        description: this.buildDescription(layer.config.name, layerType),
        stage: this.inferStage(layer.config.name, layerType)
      },
      outputShape
    };
  }

  private computeModelOutputShape(
    layer: ModelTopologyLayer,
    inputShape: number[],
    layerType: ModeCNetworkLayer['type']
  ): number[] {
    if (layerType === 'conv') {
      const kernel = layer.config.kernel_size?.[0] ?? 3;
      const stride = layer.config.strides?.[0] ?? 1;
      const outputHeight = Math.floor((inputShape[0] - kernel) / stride) + 1;
      const outputWidth = Math.floor((inputShape[1] - kernel) / stride) + 1;
      return [outputHeight, outputWidth, layer.config.filters ?? inputShape[2] ?? 0];
    }

    if (layerType === 'relu') {
      return [...inputShape];
    }

    if (layerType === 'pool') {
      const pool = layer.config.pool_size?.[0] ?? 2;
      const stride = layer.config.strides?.[0] ?? pool;
      const outputHeight = Math.floor((inputShape[0] - pool) / stride) + 1;
      const outputWidth = Math.floor((inputShape[1] - pool) / stride) + 1;
      return [outputHeight, outputWidth, inputShape[2] ?? 0];
    }

    if (layerType === 'flatten') {
      return [(inputShape[0] ?? 1) * (inputShape[1] ?? 1) * (inputShape[2] ?? 1)];
    }

    return [layer.config.units ?? inputShape[0] ?? 0];
  }

  private computeModelParameterCount(
    layer: ModelTopologyLayer,
    inputShape: number[],
    outputShape: number[],
    layerType: ModeCNetworkLayer['type']
  ): number {
    if (layerType === 'conv') {
      const kernelHeight = layer.config.kernel_size?.[0] ?? 3;
      const kernelWidth = layer.config.kernel_size?.[1] ?? kernelHeight;
      const inputChannels = inputShape[2] ?? 0;
      const outputChannels = outputShape[2] ?? 0;
      return kernelHeight * kernelWidth * inputChannels * outputChannels + outputChannels;
    }

    if (layerType === 'output') {
      const inputUnits = inputShape[0] ?? 0;
      const outputUnits = outputShape[0] ?? 0;
      return inputUnits * outputUnits + outputUnits;
    }

    return 0;
  }

  private inferModelKernelSize(
    layer: ModelTopologyLayer,
    layerType: ModeCNetworkLayer['type']
  ): number | null {
    if (layerType === 'conv') {
      return layer.config.kernel_size?.[0] ?? null;
    }
    if (layerType === 'pool') {
      return layer.config.pool_size?.[0] ?? null;
    }
    return null;
  }

  private inferLayerType(name: string): ModeCNetworkLayer['type'] {
    if (name.includes('conv')) return 'conv';
    if (name.includes('relu')) return 'relu';
    if (name.includes('pool')) return 'pool';
    if (name.includes('flatten')) return 'flatten';
    if (name.includes('input')) return 'input';
    return 'output';
  }

  private inferStage(name: string, type: ModeCNetworkLayer['type']): ModeCNetworkLayer['stage'] {
    if (type === 'flatten') return 'bridge';
    if (type === 'output') return 'classifier';
    if (name.includes('_2_')) return 'encoder-b';
    return 'encoder-a';
  }

  private buildLayerTitle(name: string, type: ModeCNetworkLayer['type']): string {
    if (type === 'flatten') return 'Flatten 层';
    if (type === 'output') return '输出层';
    const humanIndex = this.extractHumanIndex(name);
    if (type === 'pool') return `最大池化 ${humanIndex}`;
    if (type === 'relu') return `ReLU ${humanIndex}`;
    return `卷积 ${humanIndex}`;
  }

  private buildShortTitle(name: string, type: ModeCNetworkLayer['type']): string {
    if (type === 'flatten') return 'Flatten';
    if (type === 'output') return '输出';
    const humanIndex = this.extractHumanIndex(name);
    if (type === 'pool') return `池化 ${humanIndex}`;
    if (type === 'relu') return `ReLU ${humanIndex}`;
    return `卷积 ${humanIndex}`;
  }

  private buildDescription(name: string, type: ModeCNetworkLayer['type']): string {
    if (type === 'input') {
      return 'RGB 输入张量是整个 CNN 推理链路的起点。';
    }
    if (type === 'conv') {
      return `${name} 会应用卷积核逐步提取更有结构的空间特征。`;
    }
    if (type === 'relu') {
      return `${name} 会引入非线性，并保留更强的正向激活。`;
    }
    if (type === 'pool') {
      return `${name} 会降低空间分辨率，同时尽量保留局部最强响应。`;
    }
    if (type === 'flatten') {
      return 'Flatten 会把最后的特征图堆栈拉平成一维向量，供分类层使用。';
    }
    return '输出层会为 10 个目标类别生成 logits，并进一步转成概率分布。';
  }

  private buildFallbackLayer(
    id: string,
    title: string,
    shortTitle: string,
    type: ModeCNetworkLayer['type'],
    channels: number,
    spatialSize: number,
    inputShapeLabel: string,
    outputShapeLabel: string,
    kernelLabel: string | null,
    parameterCount: number,
    description: string,
    stage: ModeCNetworkLayer['stage']
  ): ModeCNetworkLayer {
    return {
      id,
      sourceName: id,
      title,
      shortTitle,
      type,
      channels,
      spatialSize,
      inputShapeLabel,
      outputShapeLabel,
      kernelLabel,
      parameterCount,
      description,
      stage
    };
  }

  private formatShape(shape: number[]): string {
    return shape.join(' x ');
  }

  private extractHumanIndex(name: string): string {
    const numbers = name.match(/\d+/g) ?? [];
    if (numbers.length === 0) return name;
    if (numbers.length === 1) return numbers[0];
    return `${numbers[0]}.${numbers[1]}`;
  }
}

export interface RawNetworkLayer {
  name: string;
  input_shape: number[];
  output_shape: number[];
  num_neurons: number;
  weights: Array<{
    bias: number;
    weights?: unknown[];
  }>;
}

export interface ModelJsonConfig {
  modelTopology: {
    model_config: {
      config: {
        layers: ModelTopologyLayer[];
      };
    };
  };
}

export interface ModelTopologyLayer {
  class_name: string;
  config: {
    name: string;
    batch_input_shape?: Array<number | null>;
    filters?: number;
    kernel_size?: number[];
    strides?: number[];
    pool_size?: number[];
    units?: number;
  };
}
