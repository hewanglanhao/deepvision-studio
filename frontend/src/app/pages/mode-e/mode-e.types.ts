export type Matrix = number[][];
export type Vector = number[];

export interface TokenEmbedding {
  token: string;
  embedding: Vector;
  position: Vector;
  x: Vector;
}

export interface AttentionHeadTrace {
  id: number;
  q: Matrix;
  k: Matrix;
  v: Matrix;
  scores: Matrix;
  weights: Matrix;
  context: Matrix;
}

export interface TransformerStep {
  id: string;
  title: string;
  formula: string;
  summary: string;
  matrixName: string;
  matrix: Matrix;
}

export interface TransformerTrace {
  tokens: string[];
  inputEmbeddings: TokenEmbedding[];
  x: Matrix;
  heads: AttentionHeadTrace[];
  multiHead: Matrix;
  projectedAttention: Matrix;
  attentionResidual: Matrix;
  attentionNorm: Matrix;
  ffnHidden: Matrix;
  ffnOutput: Matrix;
  blockOutput: Matrix;
  steps: TransformerStep[];
  parameterCount: number;
}

export interface TransformerPreset {
  id: string;
  label: string;
  text: string;
}
