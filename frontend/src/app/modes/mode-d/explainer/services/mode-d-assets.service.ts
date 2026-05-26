import { Injectable } from '@angular/core';
import { ModeDExample } from '../models/mode-d.types';

@Injectable({ providedIn: 'root' })
export class ModeDAssetsService {
  readonly examples: ModeDExample[] = [
    {
      id: 'story',
      title: '叙事补全',
      subtitle: '观察模型如何沿着局部上下文继续补全句子',
      text: 'the cat sat on the',
      focus: '这类短句通常会让模型优先回看最近的名词和介词结构，再决定最自然的续写词。',
      candidateTokens: ['mat', 'floor', 'chair', 'sofa', 'window']
    },
    {
      id: 'research',
      title: '技术表达',
      subtitle: '观察概念词和后续术语之间的依赖',
      text: 'attention helps the model learn',
      focus: '技术句子里，attention head 往往会回看关键概念词，并把注意力集中在决定后续术语的上下文上。',
      candidateTokens: ['context', 'patterns', 'relations', 'weights', 'tokens']
    },
    {
      id: 'dialogue',
      title: '解释式续写',
      subtitle: '查看工具词、名词短语和补语之间的关联',
      text: 'we can explain this transformer with',
      focus: '当句子已经形成“动词 + with”这类结构时，模型通常会优先预测工具词、示例词或解释性名词短语。',
      candidateTokens: ['attention', 'examples', 'tokens', 'heatmaps', 'details']
    }
  ];

  readonly blockOptions = Array.from({ length: 12 }, (_, index) => ({
    id: index,
    label: `第 ${index + 1} 层`
  }));

  readonly headOptions = Array.from({ length: 12 }, (_, index) => ({
    id: index,
    label: `第 ${index + 1} 头`
  }));
}
