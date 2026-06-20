import { Injectable } from '@angular/core';
import type { ModeFDatasetPreset, ModeFNetworkPreset, ModeFSequenceSample } from '../models/mode-f.types';

function mkSample(id: number, inputs: number[][], label: number, labelName?: string): ModeFSequenceSample {
  return { id, inputs, label, labelName };
}

/** One-hot encoding: 1 → [1,0], 0 → [0,1] */
function enc(v: number): number[] { return v ? [1, 0] : [0, 1]; }
const BLANK = [0, 0];

// --- synthetic datasets ---

/** Remember a single bit across 3 blank steps */
function echoDataset(n: number): ModeFSequenceSample[] {
  const samples: ModeFSequenceSample[] = [];
  for (let i = 0; i < n; i++) {
    const v = Math.random() > 0.5 ? 1 : 0;
    const seq = [enc(v), BLANK, BLANK, BLANK];
    samples.push(mkSample(i, seq, v, v === 0 ? 'A' : 'B'));
  }
  return samples;
}

/** XOR of two bits given at adjacent steps */
function xorMemoryDataset(n: number): ModeFSequenceSample[] {
  const samples: ModeFSequenceSample[] = [];
  for (let i = 0; i < n; i++) {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;
    const seq = [enc(a), enc(b), BLANK, BLANK];
    samples.push(mkSample(i, seq, a ^ b, (a ^ b) === 0 ? '相同' : '不同'));
  }
  return samples;
}

/** Compare first and last bit across two blank steps — harder than echo */
function delayMatchDataset(n: number): ModeFSequenceSample[] {
  const samples: ModeFSequenceSample[] = [];
  for (let i = 0; i < n; i++) {
    const a = Math.random() > 0.5 ? 1 : 0;
    const b = Math.random() > 0.5 ? 1 : 0;
    const seq = [enc(a), BLANK, BLANK, enc(b)];
    const label = a === b ? 1 : 0; // 1 = same, 0 = different
    samples.push(mkSample(i, seq, label, label === 1 ? '相同' : '不同'));
  }
  return samples;
}

@Injectable({ providedIn: 'root' })
export class ModeFAssetsService {
  readonly datasetPresets: ModeFDatasetPreset[] = [
    {
      id: 'echo', name: '延迟记忆',
      description: '第一步给出一个 bit，经过三步空白后在最后一步输出分类。测试 RNN 能否将信息跨时间步保留。',
      samples: echoDataset(200), inputDim: 2, hiddenDim: 4, outputDim: 2,
      classLabels: ['A (bit=0)', 'B (bit=1)'], maxTimeSteps: 4,
    },
    {
      id: 'memory', name: 'XOR 记忆',
      description: '前两步各给一个 bit，后两步空白，最后一步判断两个 bit 是否相同。测试 RNN 的记忆与组合判断能力。',
      samples: xorMemoryDataset(200), inputDim: 2, hiddenDim: 8, outputDim: 2,
      classLabels: ['相同', '不同'], maxTimeSteps: 4,
    },
    {
      id: 'delay-match', name: '延迟对比',
      description: '第一步和第四步各给一个 bit，中间两步空白。判断首尾 bit 是否相同。需要跨两步记忆首个 bit 再与末尾比较，难度最高。',
      samples: delayMatchDataset(200), inputDim: 2, hiddenDim: 6, outputDim: 2,
      classLabels: ['相同', '不同'], maxTimeSteps: 4,
    },
  ];

  readonly networkPresets: ModeFNetworkPreset[] = [
    {
      id: 'echo-simple', name: '延迟记忆 RNN',
      description: '小 RNN 学习延迟记忆', cellType: 'tanh',
      inputDim: 2, hiddenDim: 4, outputDim: 2, datasetId: 'echo',
    },
    {
      id: 'memory-rnn', name: 'XOR 记忆 RNN',
      description: 'RNN 学习 XOR 记忆', cellType: 'tanh',
      inputDim: 2, hiddenDim: 8, outputDim: 2, datasetId: 'memory',
    },
    {
      id: 'delay-match-rnn', name: '延迟对比 RNN',
      description: 'RNN 学习延迟对比', cellType: 'tanh',
      inputDim: 2, hiddenDim: 6, outputDim: 2, datasetId: 'delay-match',
    },
  ];
}
