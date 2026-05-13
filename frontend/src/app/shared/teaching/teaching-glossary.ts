export interface TeachingTerm {
  id: string;
  title: string;
  aliases: string[];
  category: '前向传播流程' | '数据与张量' | '网络层' | '层参数' | '输出解读';
  summary: string;
  details: string[];
  mode: 'A';
}

export const MODE_A_TEACHING_TERMS: TeachingTerm[] = [
  {
    id: 'forward-pass',
    title: '前向传播',
    aliases: ['前向传播实验室', '前向结果'],
    category: '前向传播流程',
    summary: '前向传播是把输入图片按网络层顺序一路计算到输出的过程。',
    details: [
      '在 A 模式中，图片先被预处理成张量，然后依次经过卷积、池化、展平、全连接、激活等层。',
      '每一层只使用当前参数做一次计算，不更新权重，所以它展示的是“模型如何看一张图”，不是训练过程。',
      '查看每层的输入 shape、输出 shape、公式和特征图，可以把抽象公式对应到真实数值变化。'
    ],
    mode: 'A'
  },
  {
    id: 'input-source',
    title: '输入来源',
    aliases: ['输入信息', '样本', '数据集'],
    category: '前向传播流程',
    summary: '输入来源决定这次前向传播使用哪一张图片。',
    details: [
      'A 模式支持从内置样本中选择图片，也可以围绕当前图片观察网络逐层输出。',
      '同一网络处理不同输入时，特征图和最终输出会改变，因为每个像素值都参与后续计算。'
    ],
    mode: 'A'
  },
  {
    id: 'preprocessing',
    title: '预处理',
    aliases: ['缩放模式', '颜色模式', '像素值取反'],
    category: '数据与张量',
    summary: '预处理把原始图片整理成网络能计算的固定张量格式。',
    details: [
      '常见预处理包括缩放尺寸、选择 RGB 或灰度、归一化到数值区间，以及必要时取反像素。',
      '预处理会改变输入张量的宽、高、通道数和数值范围，后续所有层都以这个张量为起点。'
    ],
    mode: 'A'
  },
  {
    id: 'normalization',
    title: '归一化',
    aliases: ['[0,1]', 'normalize'],
    category: '数据与张量',
    summary: '归一化把像素值压到更稳定的数值范围，方便网络层计算。',
    details: [
      '图片原始像素通常是 0 到 255，归一化后常变成 0 到 1。',
      '数值范围稳定后，卷积、激活和全连接层的输出更容易观察，也更接近真实模型输入习惯。'
    ],
    mode: 'A'
  },
  {
    id: 'shape',
    title: 'Shape',
    aliases: ['实际计算尺寸', '输入形状', '输出形状'],
    category: '数据与张量',
    summary: 'Shape 描述张量的尺寸，例如 高 × 宽 × 通道。',
    details: [
      '图像张量常写成 H × W × C，分别表示高度、宽度和通道数。',
      '卷积、池化、展平等层都会改变 shape。观察 shape 路径能帮助判断网络结构是否合理。'
    ],
    mode: 'A'
  },
  {
    id: 'tensor',
    title: '张量',
    aliases: ['图像张量', '输入张量', '输出张量'],
    category: '数据与张量',
    summary: '张量是网络内部传递的数据容器，可以理解为多维数组。',
    details: [
      '一张 RGB 图片可以看作三维张量：高度、宽度、颜色通道。',
      '全连接层之后的结果常变成一维向量，也仍然是张量。A 模式会把张量转成图像、通道图或数值统计辅助观察。'
    ],
    mode: 'A'
  },
  {
    id: 'channel',
    title: '通道',
    aliases: ['输出通道', '输入通道', '特征图通道'],
    category: '数据与张量',
    summary: '通道是一组并行的特征维度。',
    details: [
      'RGB 图片有 3 个颜色通道；卷积层输出通道通常表示不同卷积核提取出的不同特征。',
      '通道越多，网络可以同时记录越多类型的边缘、纹理或局部模式，但计算量和参数量也会上升。'
    ],
    mode: 'A'
  },
  {
    id: 'network-template',
    title: '网络模板',
    aliases: ['选择模板', '网络结构'],
    category: '网络层',
    summary: '网络模板是一组预设层结构，用来快速搭出可运行的前向传播网络。',
    details: [
      '模板通常包含输入层、若干特征提取层和输出层。',
      '你可以基于模板增删层、调整参数，再观察 shape 和输出怎样变化。'
    ],
    mode: 'A'
  },
  {
    id: 'conv2d',
    title: '卷积层',
    aliases: ['Conv', '卷积核矩阵', '经典卷积核预设'],
    category: '网络层',
    summary: '卷积层用小矩阵在图像上滑动，提取局部特征。',
    details: [
      '卷积核会在每个局部窗口中做加权求和，生成新的特征图。',
      '不同卷积核会强调不同模式，例如边缘、锐化、模糊或某个方向的变化。'
    ],
    mode: 'A'
  },
  {
    id: 'kernel-size',
    title: '核大小',
    aliases: ['kernel size'],
    category: '层参数',
    summary: '核大小决定卷积或池化一次看多大的局部区域。',
    details: [
      '3 × 3 核每次看中心像素周围一圈，5 × 5 核能看更大范围。',
      '核越大，感受区域越宽，但参数和计算量通常也越高。'
    ],
    mode: 'A'
  },
  {
    id: 'stride',
    title: 'Stride',
    aliases: ['步幅'],
    category: '层参数',
    summary: 'Stride 是卷积核或池化窗口每次移动的步长。',
    details: [
      'stride 为 1 时逐格滑动，输出保留更多空间细节。',
      'stride 变大时输出尺寸会变小，计算更少，但可能丢失细节。'
    ],
    mode: 'A'
  },
  {
    id: 'padding',
    title: 'Padding',
    aliases: ['填充'],
    category: '层参数',
    summary: 'Padding 会在输入边缘补值，控制边界信息和输出尺寸。',
    details: [
      '没有 padding 时，卷积窗口无法完整覆盖边缘，输出尺寸通常变小。',
      '适当 padding 可以让输出宽高更接近输入，并让边缘像素也参与足够多的计算。'
    ],
    mode: 'A'
  },
  {
    id: 'dilation',
    title: 'Dilation',
    aliases: ['空洞卷积'],
    category: '层参数',
    summary: 'Dilation 会拉开卷积核采样点，让同样大小的核看到更大范围。',
    details: [
      'dilation 为 1 是普通卷积；更大的 dilation 会跳格采样。',
      '它能扩大感受野，但也可能让输出更稀疏、更难直观解释。'
    ],
    mode: 'A'
  },
  {
    id: 'bias',
    title: 'Bias',
    aliases: ['偏置'],
    category: '层参数',
    summary: 'Bias 是加在加权求和结果上的可调偏移量。',
    details: [
      '卷积层和全连接层常在求和后加 bias，再进入激活函数。',
      '它可以整体抬高或压低某个输出通道、神经元或类别的响应。'
    ],
    mode: 'A'
  },
  {
    id: 'activation',
    title: '激活函数',
    aliases: ['ReLU', 'Sigmoid', 'Tanh', 'GELU'],
    category: '网络层',
    summary: '激活函数给网络加入非线性，让模型不只是做线性变换。',
    details: [
      '如果没有激活函数，多层线性计算叠在一起仍然等价于一层线性计算，表达能力有限。',
      'ReLU 会把负数截为 0；Sigmoid 会压到 0 到 1；Tanh 会压到 -1 到 1；GELU 常用于更平滑的深层网络。'
    ],
    mode: 'A'
  },
  {
    id: 'pooling',
    title: '池化层',
    aliases: ['池化方式', '最大池化', '平均池化'],
    category: '网络层',
    summary: '池化层把局部区域压缩成一个值，降低空间尺寸。',
    details: [
      '最大池化取窗口中的最大值，常保留最强响应。',
      '平均池化取窗口平均值，常让结果更平滑。池化能降低计算量，也让特征对小位移更稳定。'
    ],
    mode: 'A'
  },
  {
    id: 'flatten',
    title: 'Flatten',
    aliases: ['展平'],
    category: '网络层',
    summary: 'Flatten 把多维特征图摊平成一维向量。',
    details: [
      '卷积和池化通常输出 H × W × C 的图像特征。',
      '全连接层需要一维输入，所以 Flatten 是卷积特征到向量分类之间的桥。'
    ],
    mode: 'A'
  },
  {
    id: 'dense',
    title: 'Dense 全连接层',
    aliases: ['Dense', '神经元数', '输出神经元'],
    category: '网络层',
    summary: 'Dense 层让每个输出神经元连接到上一层的全部输入。',
    details: [
      '它会把输入向量和权重矩阵相乘，再加上 bias。',
      'Dense 层常用于把前面提取到的特征组合成更抽象的判断。'
    ],
    mode: 'A'
  },
  {
    id: 'dropout',
    title: 'Dropout',
    aliases: ['随机丢弃', 'Dropout 比率'],
    category: '网络层',
    summary: 'Dropout 在训练时随机屏蔽部分神经元，减少过拟合。',
    details: [
      'A 模式主要展示前向传播，因此默认更适合关闭随机丢弃，让结果稳定可复现。',
      '启用演示后，部分值会被置零，你可以观察随机性如何影响后续输出。'
    ],
    mode: 'A'
  },
  {
    id: 'layer-inspector',
    title: '层检查器',
    aliases: ['公式', '可视化输出', '层参数'],
    category: '前向传播流程',
    summary: '层检查器聚焦当前选中层，展示它的输入、输出、公式和统计。',
    details: [
      '它把网络图中的某一层展开成可解释面板，适合逐层回答“这一层做了什么”。',
      '结合公式、特征图和统计值看，能避免只看结构名而不知道实际计算效果。'
    ],
    mode: 'A'
  },
  {
    id: 'feature-map',
    title: '特征图',
    aliases: ['可视化输出', '输出特征图通道'],
    category: '输出解读',
    summary: '特征图是卷积层等图像型层输出的空间响应图。',
    details: [
      '亮的区域通常表示对应卷积核或通道在这些位置响应较强。',
      '不同通道的特征图可以关注不同方向、纹理或局部形状。'
    ],
    mode: 'A'
  },
  {
    id: 'tensor-stats',
    title: '张量统计',
    aliases: ['最小值', '最大值', '均值', '非零率'],
    category: '输出解读',
    summary: '张量统计用几个数字概括当前输出的数值分布。',
    details: [
      '最小值、最大值、均值能帮助判断输出是否过大、过小或几乎没有变化。',
      '非零率常用于观察 ReLU、Dropout 等操作后还有多少位置保留了有效响应。'
    ],
    mode: 'A'
  },
  {
    id: 'top-k',
    title: 'Top-K',
    aliases: ['输出分布', '输出类别'],
    category: '输出解读',
    summary: 'Top-K 展示最终输出中分数最高的几个类别。',
    details: [
      '它适合快速看模型当前最倾向的预测结果。',
      'A 模式里的 Top-K 更强调前向输出分布的教学解释，不代表模型已经通过训练获得真实识别能力。'
    ],
    mode: 'A'
  }
];

export const TEACHING_TERMS = MODE_A_TEACHING_TERMS;

export function findTeachingTerm(id: string): TeachingTerm | undefined {
  return TEACHING_TERMS.find(term => term.id === id);
}
