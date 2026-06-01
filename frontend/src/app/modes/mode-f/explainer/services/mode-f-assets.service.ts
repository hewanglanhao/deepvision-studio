import { Injectable } from '@angular/core';
import type { ModeFDatasetPreset, ModeFNetworkPreset, ModeFSequenceSample } from '../models/mode-f.types';

function mkSample(id: number, inputs: number[][], label: number, labelName?: string): ModeFSequenceSample {
  return { id, inputs, label, labelName };
}

// --- synthetic datasets ---

function echoDataset(n: number): ModeFSequenceSample[] {
  const samples: ModeFSequenceSample[] = [];
  for (let i = 0; i < n; i++) {
    const v = Math.random() > 0.5 ? 1 : 0;
    const seq = [[v, 0], [0, 0], [0, 0], [0, 0]];
    // Task: after seeing input [v,0] at step 0, output class v at step 3
    // 3 zeros then output. Class = the value seen at step 0.
    const label = v;
    samples.push(mkSample(i, seq, label, label === 0 ? 'A' : 'B'));
  }
  return samples;
}

function memoryDataset(n: number): ModeFSequenceSample[] {
  const samples: ModeFSequenceSample[] = [];
  for (let i = 0; i < n; i++) {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;
    const seq = [[a, b], [0, 0], [0, 0], [0, 0]];
    // Task: classify based on both values seen: a XOR b
    const label = (a ^ b);
    samples.push(mkSample(i, seq, label, label === 0 ? '相同' : '不同'));
  }
  return samples;
}

function alternationDataset(n: number): ModeFSequenceSample[] {
  const samples: ModeFSequenceSample[] = [];
  for (let i = 0; i < n; i++) {
    const v1 = Math.random() > 0.5 ? 1 : 0;
    const v2 = Math.random() > 0.5 ? 1 : 0;
    const seq = [[v1, 0], [v2, 0], [0, 0], [0, 0]];
    // Task: detect if sequence alternated. 1 if v1 != v2, 0 if v1 == v2
    const label = v1 !== v2 ? 1 : 0;
    samples.push(mkSample(i, seq, label, label === 0 ? '相同' : '交替'));
  }
  return samples;
}

@Injectable({ providedIn: 'root' })
export class ModeFAssetsService {
  readonly datasetPresets: ModeFDatasetPreset[] = [
    { id: 'echo', name: '延迟记忆', description: '看到第一个输入后，经过几步延迟再输出分类。测试 RNN 能否"记住"早期信息。', samples: echoDataset(200), inputDim: 2, hiddenDim: 4, outputDim: 2, classLabels: ['A', 'B'], maxTimeSteps: 4 },
    { id: 'memory', name: 'XOR 记忆', description: '前两步各给一个 bit，最后一步判断两个 bit 是否相同（等价于 XOR）', samples: memoryDataset(200), inputDim: 2, hiddenDim: 4, outputDim: 2, classLabels: ['相同', '不同'], maxTimeSteps: 4 },
    { id: 'alternation', name: '交替检测', description: '前两步各给一个 bit，检测序列是否发生了交替。需要 RNN 比较相邻时间步的输入。', samples: alternationDataset(200), inputDim: 2, hiddenDim: 6, outputDim: 2, classLabels: ['相同', '交替'], maxTimeSteps: 4 },
  ];

  readonly networkPresets: ModeFNetworkPreset[] = [
    { id: 'echo-simple', name: '延迟记忆 RNN', description: '小 RNN 学习延迟记忆', cellType: 'tanh', inputDim: 2, hiddenDim: 4, outputDim: 2, datasetId: 'echo' },
    { id: 'memory-rnn', name: 'XOR 记忆 RNN', description: 'RNN 学习 XOR 记忆', cellType: 'tanh', inputDim: 2, hiddenDim: 4, outputDim: 2, datasetId: 'memory' },
    { id: 'alternation-rnn', name: '交替检测 RNN', description: 'RNN 检测序列交替', cellType: 'tanh', inputDim: 2, hiddenDim: 6, outputDim: 2, datasetId: 'alternation' },
  ];
}
