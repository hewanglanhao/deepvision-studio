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
  readonly modelDataUrl = '/modules/cnn-explainer/assets/data/model.json';

  readonly sampleOptions: ModeCSampleOption[] = [
    {
      id: 'espresso',
      title: '浓缩咖啡',
      label: '咖啡纹理样例',
      description: '默认教学样例，用来验证原生渲染外壳以及真实推理链是否工作正常。',
      assetPath: '/modules/cnn-explainer/assets/img/espresso_1.jpeg',
      predictedClass: '',
      confidence: 0,
      topClasses: []
    },
    {
      id: 'panda',
      title: '熊猫',
      label: '高对比度动物样例',
      description: '适合观察类别概率展示以及中间激活图的解释效果。',
      assetPath: '/modules/cnn-explainer/assets/img/panda_1.jpeg',
      predictedClass: '',
      confidence: 0,
      topClasses: []
    },
    {
      id: 'pizza',
      title: '披萨',
      label: '食物类别样例',
      description: '语义目标较明确，适合验证图像切换后的联动行为。',
      assetPath: '/modules/cnn-explainer/assets/img/pizza_1.jpeg',
      predictedClass: '',
      confidence: 0,
      topClasses: []
    },
    {
      id: 'bus',
      title: '公交车',
      label: '交通工具类别样例',
      description: '适合测试在输入差异较大的情况下，总览图更新是否仍然稳定。',
      assetPath: '/modules/cnn-explainer/assets/img/bus_1.jpeg',
      predictedClass: '',
      confidence: 0,
      topClasses: []
    }
  ];

  readonly fallbackNetworkLayers: ModeCNetworkLayer[] = [
    this.buildFallbackLayer('input', '输入图像', '输入', 'input', 3, 64, '64 x 64 x 3', '64 x 64 x 3', null, 0, 'RGB 图像会先被裁剪并标准化到 64×64，然后再送入 CNN。', 'encoder-a'),
    this.buildFallbackLayer('conv_1_1', '卷积 1.1', '卷积 1.1', 'conv', 10, 62, '64 x 64 x 3', '62 x 62 x 10', '3 x 3', 280, '第一层空间特征提取层，负责检测边缘、斑块和基础纹理。', 'encoder-a'),
    this.buildFallbackLayer('relu_1_1', 'ReLU 1.1', 'ReLU 1.1', 'relu', 10, 62, '62 x 62 x 10', '62 x 62 x 10', null, 0, '引入非线性，并压制第一组卷积响应中的负值部分。', 'encoder-a'),
    this.buildFallbackLayer('conv_1_2', '卷积 1.2', '卷积 1.2', 'conv', 10, 60, '62 x 62 x 10', '60 x 60 x 10', '3 x 3', 910, '在第一组激活结果的基础上，进一步组合出更结构化的局部特征。', 'encoder-a'),
    this.buildFallbackLayer('relu_1_2', 'ReLU 1.2', 'ReLU 1.2', 'relu', 10, 60, '60 x 60 x 10', '60 x 60 x 10', null, 0, '保留更具判别力的激活响应，并为第一次下采样做准备。', 'encoder-a'),
    this.buildFallbackLayer('max_pool_1', '最大池化 1', '池化 1', 'pool', 10, 30, '60 x 60 x 10', '30 x 30 x 10', '2 x 2', 0, '降低特征图分辨率，同时尽量保留最强响应区域。', 'encoder-a'),
    this.buildFallbackLayer('conv_2_1', '卷积 2.1', '卷积 2.1', 'conv', 10, 28, '30 x 30 x 10', '28 x 28 x 10', '3 x 3', 910, '在更紧凑的空间尺度上继续提取组合特征。', 'encoder-b'),
    this.buildFallbackLayer('relu_2_1', 'ReLU 2.1', 'ReLU 2.1', 'relu', 10, 28, '28 x 28 x 10', '28 x 28 x 10', null, 0, '继续抑制负响应，保留更稳定的正向证据。', 'encoder-b'),
    this.buildFallbackLayer('conv_2_2', '卷积 2.2', '卷积 2.2', 'conv', 10, 26, '28 x 28 x 10', '26 x 26 x 10', '3 x 3', 910, '进一步巩固类别相关的局部特征模式。', 'encoder-b'),
    this.buildFallbackLayer('relu_2_2', 'ReLU 2.2', 'ReLU 2.2', 'relu', 10, 26, '26 x 26 x 10', '26 x 26 x 10', null, 0, '在进入第二次池化之前保留最后一层正向激活。', 'encoder-b'),
    this.buildFallbackLayer('max_pool_2', '最大池化 2', '池化 2', 'pool', 10, 13, '26 x 26 x 10', '13 x 13 x 10', '2 x 2', 0, '再次压缩空间分辨率，为分类器提供更紧凑的特征表示。', 'encoder-b'),
    this.buildFallbackLayer('flatten', 'Flatten 层', 'Flatten', 'flatten', 1690, 1, '13 x 13 x 10', '1690', null, 0, '将最后的特征堆栈展开成一维向量，供分类层使用。', 'bridge'),
    this.buildFallbackLayer('output', '输出层', '输出', 'output', 10, 1, '1690', '10', null, 16910, '对 10 个训练类别生成 dense logits，随后再解释为最终类别概率。', 'classifier')
  ];

  readonly overviewStages: ModeCOverviewStage[] = [
    { id: 'input', title: '输入样例', summary: '原生 Angular 版本中的图像选择与预处理入口。', status: 'ready' },
    { id: 'graph', title: '网络总览', summary: '原生 Angular 的 SVG 总览已经可用，后续可继续承载更丰富的激活可视化。', status: 'ready' },
    { id: 'detail', title: '细节面板', summary: '承载卷积、激活、池化和 softmax 的分步解释。', status: 'in-progress' },
    { id: 'article', title: '教学文章', summary: '用结构化内容卡片替代原来冗长的单页文章区域。', status: 'planned' }
  ];

  readonly detailTopics: ModeCDetailTopic[] = [
    { id: 'overview-graph', title: '总览图', description: '定义原生 Angular 图形的结构约定、布局槽位和交互状态。', priority: 'P0' },
    { id: 'sample-switching', title: '样例切换', description: '在预设教学样例间切换，并保持平台原生的选中状态管理。', priority: 'P0' },
    { id: 'conv-panel', title: '卷积面板', description: '在总览图 MVP 建好之后，优先深入实现的首个解释面板。', priority: 'P1' },
    { id: 'softmax-panel', title: 'Softmax 面板', description: '用于展示概率解释和最终分类结果的说明界面。', priority: 'P1' }
  ];

  readonly milestones: ModeCMilestone[] = [
    { id: 'shell', title: '原生外壳已上线', note: 'Mode C 的顶层体验已经不再依赖 iframe。', status: 'ready' },
    { id: 'state', title: '类型化状态服务', note: '共享 UI 状态已经改为用 Angular 服务管理，而不是 Svelte store。', status: 'ready' },
    { id: 'graph', title: '总览图迁移', note: 'Angular 外壳已经能原生渲染总览图，并支持层选择和样例预测联动。', status: 'ready' },
    { id: 'detail-ready', title: '细节联动接线', note: '总览图中选中的层已经可以驱动右侧细节面板的上下文切换。', status: 'in-progress' }
  ];

  readonly articleSections: ModeCArticleSection[] = [
    {
      id: 'goal',
      eyebrow: 'Mode C 重写',
      title: '原生外壳的作用',
      body: [
        '这一版原生 Angular 切片首先为 DeepVision Studio 中的 Mode C 建立了长期稳定的宿主结构。',
        '在图形逻辑和教学逻辑进一步迁移之前，它先提供稳定外壳、类型化状态以及与平台一致的界面基础。'
      ]
    },
    {
      id: 'mapping',
      eyebrow: '结构映射',
      title: '旧应用如何迁移到 Angular',
      body: [
        '原始 Svelte 应用把模型加载、全局状态、文章内容和 D3 交互都混在少量大文件中。',
        '重写之后，这些职责被拆分为功能组件、服务和类型化模型，从而让模块能够像平台其他部分一样持续演进。'
      ],
      bullets: [
        '总览图变成独立的功能组件',
        '细节视图变成具备明确输入的面板组件',
        '文章内容改造成结构化内容块，而不是一整段超长模板'
      ]
    },
    {
      id: 'next',
      eyebrow: '下一步',
      title: '第二阶段将替换什么',
      body: [
        '当前这个占位式总览区域是有意保持静态的，它标记出了后续 D3 或 SVG 图形重写的精确承载面。',
        '当第二阶段开始时，我们可以直接替换内部实现，而不必再次重做页面外壳。'
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
        'RGB 图像会先被裁剪并标准化到 64×64，然后再送入 CNN。',
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
      return 'RGB 输入张量，是整个 CNN 处理链条的起点。';
    }
    if (type === 'conv') {
      return `${name} 会应用学习得到的空间卷积核，逐步检测更有结构的视觉模式。`;
    }
    if (type === 'relu') {
      return `${name} 会引入非线性，并保留更强的正向激活。`;
    }
    if (type === 'pool') {
      return `${name} 会降低空间分辨率，同时尽量保留局部最强响应。`;
    }
    if (type === 'flatten') {
      return 'Flatten 会把最后的特征堆栈拉平成一维向量，供分类层使用。';
    }
    return '输出层会为 10 个目标类别生成最终 logits，并进一步转成概率分布。';
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
