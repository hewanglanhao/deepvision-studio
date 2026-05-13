import type {
  AttentionHeadTrace,
  Matrix,
  TokenEmbedding,
  TransformerTrace,
  Vector,
} from './mode-e.types';

const D_MODEL = 4;
const D_HEAD = 2;

const VOCAB: Record<string, Vector> = {
  i: [0.42, -0.18, 0.31, 0.09],
  study: [0.55, 0.24, -0.12, 0.38],
  transformer: [0.62, -0.33, 0.48, 0.51],
  attention: [0.47, 0.58, -0.21, 0.44],
  learns: [0.36, 0.46, 0.28, -0.19],
  context: [0.24, 0.66, -0.35, 0.33],
  tokens: [0.51, -0.27, 0.17, 0.61],
  share: [0.18, 0.52, 0.41, -0.28],
  meaning: [0.44, 0.37, -0.31, 0.57],
  mask: [0.29, -0.41, 0.63, 0.22],
  future: [0.35, -0.52, 0.54, 0.18],
  words: [0.49, 0.12, -0.24, 0.46],
  build: [0.31, 0.43, 0.32, -0.11],
  relation: [0.27, 0.61, -0.18, 0.39],
  unk: [0.12, -0.08, 0.16, 0.05],
};

const WQ: Matrix[] = [
  [
    [0.60, -0.20],
    [0.10, 0.50],
    [-0.35, 0.25],
    [0.40, 0.15],
  ],
  [
    [-0.25, 0.45],
    [0.55, 0.10],
    [0.30, -0.40],
    [0.12, 0.52],
  ],
];

const WK: Matrix[] = [
  [
    [0.35, 0.28],
    [-0.42, 0.30],
    [0.56, -0.18],
    [0.21, 0.44],
  ],
  [
    [0.42, -0.24],
    [0.18, 0.58],
    [-0.38, 0.33],
    [0.47, 0.12],
  ],
];

const WV: Matrix[] = [
  [
    [0.45, -0.12],
    [0.22, 0.36],
    [-0.28, 0.41],
    [0.31, 0.18],
  ],
  [
    [0.18, 0.53],
    [0.44, -0.16],
    [0.27, 0.39],
    [-0.35, 0.24],
  ],
];

const WO: Matrix = [
  [0.42, -0.18, 0.33, 0.21],
  [0.25, 0.36, -0.22, 0.44],
  [-0.31, 0.28, 0.49, -0.15],
  [0.19, 0.41, 0.16, 0.37],
];

const W1: Matrix = [
  [0.32, -0.21, 0.41, 0.18, -0.30, 0.25, 0.13, 0.36],
  [0.27, 0.43, -0.18, 0.22, 0.31, -0.26, 0.39, 0.12],
  [-0.35, 0.16, 0.29, -0.42, 0.24, 0.33, -0.14, 0.45],
  [0.21, 0.38, 0.11, 0.34, -0.28, 0.17, 0.42, -0.20],
];

const B1: Vector = [0.03, -0.02, 0.01, 0.04, -0.01, 0.02, 0.00, 0.03];

const W2: Matrix = [
  [0.24, -0.18, 0.33, 0.15],
  [0.41, 0.21, -0.16, 0.28],
  [-0.22, 0.35, 0.19, -0.31],
  [0.17, 0.26, 0.42, 0.11],
  [0.38, -0.27, 0.14, 0.32],
  [-0.13, 0.44, 0.25, -0.20],
  [0.29, 0.12, -0.36, 0.40],
  [0.16, -0.33, 0.31, 0.23],
];

const B2: Vector = [0.02, -0.01, 0.03, 0.00];

export const MODE_E_VOCABULARY = Object.keys(VOCAB).filter(token => token !== 'unk');

export function runTransformerBlock(text: string, causalMask: boolean): TransformerTrace {
  const tokens = tokenize(text);
  const embedded = tokens.map((token, index) => embedToken(token, index));
  const x = embedded.map(item => item.x);
  const heads = WQ.map((wq, index) => computeHead(index, x, wq, WK[index]!, WV[index]!, causalMask));
  const multiHead = concatHeads(heads.map(head => head.context));
  const projectedAttention = matMul(multiHead, WO);
  const attentionResidual = add(x, projectedAttention);
  const attentionNorm = layerNorm(attentionResidual);
  const ffnHiddenLinear = addBias(matMul(attentionNorm, W1), B1);
  const ffnHidden = ffnHiddenLinear.map(row => row.map(gelu));
  const ffnOutput = addBias(matMul(ffnHidden, W2), B2);
  const blockOutput = layerNorm(add(attentionNorm, ffnOutput));

  return {
    tokens,
    inputEmbeddings: embedded,
    x: roundMatrix(x),
    heads: heads.map(roundHead),
    multiHead: roundMatrix(multiHead),
    projectedAttention: roundMatrix(projectedAttention),
    attentionResidual: roundMatrix(attentionResidual),
    attentionNorm: roundMatrix(attentionNorm),
    ffnHidden: roundMatrix(ffnHidden),
    ffnOutput: roundMatrix(ffnOutput),
    blockOutput: roundMatrix(blockOutput),
    parameterCount: countParameters(),
    steps: [
      {
        id: 'embed',
        title: 'Token + 位置编码',
        formula: 'X = E(token) + P(pos)',
        summary: '每个 token 先查表得到词向量，再加上正弦位置编码，形成 Transformer 的输入矩阵 X。',
        matrixName: 'X',
        matrix: roundMatrix(x),
      },
      {
        id: 'qkv',
        title: '生成 Q / K / V',
        formula: 'Q = XWq, K = XWk, V = XWv',
        summary: '每个注意力头都有独立的 Wq、Wk、Wv。这里展示当前选中头的真实矩阵乘法结果。',
        matrixName: 'Q(head 1)',
        matrix: roundMatrix(heads[0]!.q),
      },
      {
        id: 'score',
        title: '缩放点积打分',
        formula: 'S = QK^T / sqrt(dk)',
        summary: 'Q 与 K 的点积衡量 token 间相关性，再除以 sqrt(dk) 稳定数值尺度。',
        matrixName: 'S(head 1)',
        matrix: roundMatrix(heads[0]!.scores),
      },
      {
        id: 'softmax',
        title: 'Softmax 注意力权重',
        formula: 'A = softmax(S)',
        summary: causalMask
          ? '每一行都是一个 token 对所有可见 token 的注意力分布；开启 mask 后，未来位置权重严格为 0。'
          : '每一行都是一个 token 对所有 token 的注意力分布，行和严格等于 1。',
        matrixName: 'A(head 1)',
        matrix: roundMatrix(heads[0]!.weights),
      },
      {
        id: 'context',
        title: '加权汇聚 Value',
        formula: 'Z = AV',
        summary: '注意力权重 A 乘以 V，把上下文信息汇聚回每个 token 的表示。',
        matrixName: 'Z(head 1)',
        matrix: roundMatrix(heads[0]!.context),
      },
      {
        id: 'multi',
        title: '多头拼接与输出投影',
        formula: 'M = Concat(Z1, Z2)Wo',
        summary: '两个头关注不同关系，拼接后再乘 Wo 回到 d_model 维度。',
        matrixName: 'M',
        matrix: roundMatrix(projectedAttention),
      },
      {
        id: 'norm1',
        title: '残差连接与 LayerNorm',
        formula: 'H = LayerNorm(X + M)',
        summary: '残差保留原始 token 表示，LayerNorm 对每个 token 的特征维做标准化。',
        matrixName: 'H',
        matrix: roundMatrix(attentionNorm),
      },
      {
        id: 'ffn',
        title: '前馈网络 FFN',
        formula: 'F = GELU(HW1 + b1)W2 + b2',
        summary: 'FFN 对每个 token 独立执行同一组非线性变换，用来提升表示能力。',
        matrixName: 'F',
        matrix: roundMatrix(ffnOutput),
      },
      {
        id: 'output',
        title: 'Transformer Block 输出',
        formula: 'Y = LayerNorm(H + F)',
        summary: '第二次残差与归一化后，得到这个 Transformer block 的最终输出。',
        matrixName: 'Y',
        matrix: roundMatrix(blockOutput),
      },
    ],
  };
}

function tokenize(text: string): string[] {
  const cleaned = text.toLowerCase().replace(/[^a-z\s]/g, ' ').trim();
  const tokens = cleaned.split(/\s+/).filter(Boolean).slice(0, 6);
  return tokens.length ? tokens : ['transformer', 'learns', 'context'];
}

function embedToken(token: string, positionIndex: number): TokenEmbedding {
  const embedding = VOCAB[token] ?? VOCAB['unk']!;
  const position = positionalEncoding(positionIndex, D_MODEL);
  return {
    token: VOCAB[token] ? token : `[unk:${token}]`,
    embedding: roundVector(embedding),
    position: roundVector(position),
    x: roundVector(embedding.map((value, index) => value + position[index]!)),
  };
}

function positionalEncoding(position: number, dimension: number): Vector {
  return Array.from({ length: dimension }, (_, i) => {
    const denominator = Math.pow(10000, (2 * Math.floor(i / 2)) / dimension);
    return i % 2 === 0 ? Math.sin(position / denominator) : Math.cos(position / denominator);
  });
}

function computeHead(
  id: number,
  x: Matrix,
  wq: Matrix,
  wk: Matrix,
  wv: Matrix,
  causalMask: boolean,
): AttentionHeadTrace {
  const q = matMul(x, wq);
  const k = matMul(x, wk);
  const v = matMul(x, wv);
  const scores = matMul(q, transpose(k)).map((row, rowIndex) =>
    row.map((value, colIndex) => {
      if (causalMask && colIndex > rowIndex) {
        return Number.NEGATIVE_INFINITY;
      }
      return value / Math.sqrt(D_HEAD);
    }),
  );
  const weights = scores.map(row => softmax(row));
  const context = matMul(weights, v);
  return { id, q, k, v, scores, weights, context };
}

function matMul(a: Matrix, b: Matrix): Matrix {
  return a.map(row =>
    b[0]!.map((_, colIndex) =>
      row.reduce((sum, value, innerIndex) => sum + value * b[innerIndex]![colIndex]!, 0),
    ),
  );
}

function add(a: Matrix, b: Matrix): Matrix {
  return a.map((row, rowIndex) => row.map((value, colIndex) => value + b[rowIndex]![colIndex]!));
}

function addBias(matrix: Matrix, bias: Vector): Matrix {
  return matrix.map(row => row.map((value, index) => value + bias[index]!));
}

function transpose(matrix: Matrix): Matrix {
  return matrix[0]!.map((_, colIndex) => matrix.map(row => row[colIndex]!));
}

function concatHeads(headMatrices: Matrix[]): Matrix {
  return headMatrices[0]!.map((_, rowIndex) => headMatrices.flatMap(matrix => matrix[rowIndex]!));
}

function softmax(row: Vector): Vector {
  const finiteValues = row.filter(Number.isFinite);
  const max = Math.max(...finiteValues);
  const exps = row.map(value => (Number.isFinite(value) ? Math.exp(value - max) : 0));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map(value => value / total);
}

function layerNorm(matrix: Matrix): Matrix {
  return matrix.map(row => {
    const mean = row.reduce((sum, value) => sum + value, 0) / row.length;
    const variance = row.reduce((sum, value) => sum + Math.pow(value - mean, 2), 0) / row.length;
    return row.map(value => (value - mean) / Math.sqrt(variance + 1e-5));
  });
}

function gelu(value: number): number {
  return 0.5 * value * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (value + 0.044715 * Math.pow(value, 3))));
}

function roundHead(head: AttentionHeadTrace): AttentionHeadTrace {
  return {
    id: head.id,
    q: roundMatrix(head.q),
    k: roundMatrix(head.k),
    v: roundMatrix(head.v),
    scores: roundMatrix(head.scores),
    weights: roundMatrix(head.weights),
    context: roundMatrix(head.context),
  };
}

function roundMatrix(matrix: Matrix): Matrix {
  return matrix.map(roundVector);
}

function roundVector(vector: Vector): Vector {
  return vector.map(value => (Number.isFinite(value) ? Number(value.toFixed(4)) : value));
}

function countParameters(): number {
  const matrices = [...WQ, ...WK, ...WV, WO, W1, W2];
  return matrices.reduce((sum, matrix) => sum + matrix.length * matrix[0]!.length, 0) + B1.length + B2.length;
}
