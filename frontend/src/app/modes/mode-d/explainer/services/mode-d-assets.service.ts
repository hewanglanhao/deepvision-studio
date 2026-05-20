import { Injectable } from '@angular/core';
import type { ModeDNetworkPreset, ModeDDatasetPreset } from '../models/mode-d.types';
import type { NetworkLayer } from '@shared/simulation/sim-models';
import { ModeDBackpropEngine } from '../engine/mode-d-backprop-engine';

// Helper to build layer IDs
let _id = 0;
function lid(): number { return ++_id; }
function resetId(): void { _id = 0; }

function makeLayer(type: string, name: string, params: Record<string, any>): any {
  return { id: lid(), type, name, inputs: [], params };
}

@Injectable({ providedIn: 'root' })
export class ModeDAssetsService {
  readonly datasetPresets: ModeDDatasetPreset[] = [];
  readonly networkPresets: ModeDNetworkPreset[] = [];

  constructor() {
    this.datasetPresets = [
      {
        id: 'xor', name: 'XOR 数据集',
        description: '四团高斯分布，构成经典的异或分类问题。线性不可分，需要至少一个隐藏层的非线性网络。',
        samples: ModeDBackpropEngine.generateXorData(400, 0.08),
        inputDim: 2, outputDim: 2, classLabels: ['蓝色类 (0)', '橙色类 (1)'],
      },
      {
        id: 'circle', name: '同心圆数据集',
        description: '内外两个同心圆，需要网络学习环形决策边界。',
        samples: ModeDBackpropEngine.generateCircleData(350, 0.12),
        inputDim: 2, outputDim: 2, classLabels: ['内圆类 (0)', '外环类 (1)'],
      },
      {
        id: 'blobs', name: '高斯团数据集',
        description: '多个高斯分布团，适合测试基本分类能力和决策边界形状。',
        samples: ModeDBackpropEngine.generateBlobData(350, [[0.25, 0.25], [0.75, 0.25], [0.5, 0.75]], 0.07),
        inputDim: 2, outputDim: 3, classLabels: ['团 A (0)', '团 B (1)', '团 C (2)'],
      },
    ];

    this.networkPresets = [
      this.buildPreset('xor-mlp', 'XOR 双层 MLP', '2 输入 → 12 隐藏 → 2 输出', 'xor', [
        makeLayer('input', '输入层', { width: 2, height: 1, channels: 1 }),
        makeLayer('dense', '隐藏层 1', { units: 12, activation: 'relu' }),
        makeLayer('output', '输出层', { units: 2, activation: 'softmax', labels: ['类 0', '类 1'] }),
      ]),
      this.buildPreset('circle-mlp', '同心圆 Sigmoid MLP', '2 输入 → 16 隐藏(Sigmoid) → 2 输出', 'circle', [
        makeLayer('input', '输入层', { width: 2, height: 1, channels: 1 }),
        makeLayer('dense', '隐藏层 1', { units: 16, activation: 'sigmoid' }),
        makeLayer('output', '输出层', { units: 2, activation: 'softmax', labels: ['内圆', '外环'] }),
      ]),
      this.buildPreset('blobs-mlp', '高斯团 MLP', '2 输入 → 4 隐藏(Sigmoid) → 3 输出', 'blobs', [
        makeLayer('input', '输入层', { width: 2, height: 1, channels: 1 }),
        makeLayer('dense', '隐藏层 1', { units: 4, activation: 'sigmoid' }),
        makeLayer('output', '输出层', { units: 3, activation: 'softmax', labels: ['团 A', '团 B', '团 C'] }),
      ]),
    ];
  }

  private buildPreset(id: string, name: string, description: string, datasetId: string, layers: any[]): ModeDNetworkPreset {
    resetId();
    const built = layers.map(l => ({ ...l, id: lid() }));
    const connections = built.slice(0, -1).map((_, i) => ({ from: built[i].id, to: built[i + 1].id }));
    return { id, name, description, layers: built as NetworkLayer[], connections, datasetId };
  }

  readonly articleSections = [
    {
      id: 'what-is-backprop', eyebrow: '核心概念', title: '什么是反向传播？',
      body: [
        '反向传播是训练神经网络的核心算法，利用链式法则从输出层开始，逐层向后计算损失函数对每个参数的梯度。',
        '过程分为四个阶段：前向传播 → 损失计算 → 反向传播 → 参数更新。',
      ],
      bullets: [
        '前向传播：输入数据经过线性变换和激活函数，得到预测输出',
        '损失计算：比较预测和真实标签，得到标量损失值',
        '反向传播：用链式法则从输出层往回计算每层参数的梯度',
        '参数更新：优化器使用梯度调整权重和偏置，使损失减小',
      ],
    },
    {
      id: 'chain-rule', eyebrow: '数学原理', title: '链式法则',
      body: [
        '链式法则是反向传播的数学基础。在神经网络中，每一层的输出是下一层的输入。',
        'Dense 层: dW = a_prevᵀ·dZ, db = Σ dZ, dA_prev = dZ·Wᵀ',
        'ReLU: dZ = dA·(Z > 0), Sigmoid: dZ = dA·σ(1-σ), Tanh: dZ = dA·(1-tanh²)',
      ],
    },
    {
      id: 'optimizers', eyebrow: '优化算法', title: '优化器比较',
      body: [
        'SGD: 简单沿负梯度方向更新。Momentum: 累积历史梯度加速收敛。Adam: 动量+自适应学习率。',
      ],
      bullets: [
        'SGD: w = w - lr·dw — 简单但可能收敛慢',
        'Momentum: v = β·v + lr·dw; w = w - v — 平滑加速',
        'Adam: 自适应学习率+动量 — 最常用，收敛最快',
      ],
    },
  ];

  readonly milestones = [
    { id: 'm1', title: '反向传播引擎', note: '纯 TypeScript 实现，零外部依赖', status: 'ready' as const },
    { id: 'm2', title: '数据集生成器', note: 'XOR、螺旋、同心圆、高斯团', status: 'ready' as const },
    { id: 'm3', title: '可视化画布', note: '网络拓扑图+梯度流动动画', status: 'ready' as const },
    { id: 'm4', title: '参数详情面板', note: '权重热力图+梯度矩阵+变化量', status: 'ready' as const },
  ];
}
