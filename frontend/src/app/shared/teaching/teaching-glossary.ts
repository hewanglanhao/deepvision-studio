export interface TeachingTerm {
  id: string;
  title: string;
  aliases: string[];
  category: string;
  summary: string;
  details: string[];
  mode: 'A' | 'B' | 'C' | 'D' | 'E' | 'F';
}

export const MODE_A_TEACHING_TERMS: TeachingTerm[] = [
  {
    id: 'forward-pass',
    title: '前向传播',
    aliases: ['forward pass', '前向过程', '推理过程'],
    category: '前向传播流程',
    summary: '前向传播是把输入样本依次送过各层，最终得到分类结果或特征响应的过程。',
    details: [
      '在 Mode A 与 Mode C 中，前向传播都强调“输入如何一步步变成输出”，只是 Mode C 更关注卷积网络内部的可解释过程。',
      '前向传播阶段不会更新权重，模型只是使用已经存在的参数完成一次计算。',
      '理解前向传播路径，有助于把层结构、特征图、输出概率和真实数值变化串起来看。'
    ],
    mode: 'A'
  },
  {
    id: 'input-source',
    title: '输入来源',
    aliases: ['样本来源', '输入样本', '数据来源'],
    category: '前向传播流程',
    summary: '输入来源说明当前前向传播所使用的是哪一个样本或图像。',
    details: [
      '不同样本会激活网络中不同的通道和区域，因此输入选择会直接影响整个解释结果。',
      '在教学场景中，固定输入来源可以帮助观察结构差异，切换输入来源则可以帮助比较模型的响应变化。'
    ],
    mode: 'A'
  },
  {
    id: 'preprocessing',
    title: '预处理',
    aliases: ['图像预处理', 'resize', 'crop'],
    category: '数据与张量',
    summary: '预处理负责把原始样本整理成模型可直接计算的输入张量。',
    details: [
      '常见步骤包括缩放、裁剪、颜色通道整理和数值归一化。',
      '如果预处理与训练时不一致，即使模型权重正确，分类结果也可能明显偏移。'
    ],
    mode: 'A'
  },
  {
    id: 'normalization',
    title: '归一化',
    aliases: ['normalize', '[0,1]', '像素缩放'],
    category: '数据与张量',
    summary: '归一化把原始像素值缩放到更稳定的数值范围，便于网络计算。',
    details: [
      '原始像素通常在 0 到 255 之间，归一化后常被压到 0 到 1 或其他更稳定的范围。',
      '归一化可以减轻不同层之间的数值震荡，让激活值和最终输出更容易解释。'
    ],
    mode: 'A'
  },
  {
    id: 'shape',
    title: 'Shape',
    aliases: ['形状', '张量形状', '输出形状'],
    category: '数据与张量',
    summary: 'Shape 描述张量的维度结构，例如高、宽和通道数。',
    details: [
      '图像张量通常写作 H × W × C，分别表示高度、宽度和通道数。',
      '每经过一层，shape 可能改变，观察 shape 路径是理解网络结构最直接的方法之一。'
    ],
    mode: 'A'
  },
  {
    id: 'tensor',
    title: '张量',
    aliases: ['tensor', '输入张量', '输出张量'],
    category: '数据与张量',
    summary: '张量是网络内部传递和计算数据的基本容器，可以理解为多维数组。',
    details: [
      '一张 RGB 图像本身就可以看作三维张量，卷积层和池化层输出的特征图也仍然是张量。',
      '从输入图像到最终分类向量，整个网络都在不断变换张量。'
    ],
    mode: 'A'
  },
  {
    id: 'channel',
    title: '通道',
    aliases: ['channel', '输出通道', '输入通道'],
    category: '数据与张量',
    summary: '通道是一组并行的特征维度，用于承载不同类型的模式响应。',
    details: [
      '输入图像通常有 3 个颜色通道；卷积层输出的多个通道则对应多个不同的卷积核响应。',
      '在 Mode C 中，通道是解释卷积网络的关键对象，因为每个通道往往关注不同的局部模式。'
    ],
    mode: 'A'
  },
  {
    id: 'network-template',
    title: '网络模板',
    aliases: ['网络结构', '模板网络', '预设结构'],
    category: '网络层',
    summary: '网络模板是一组预设好的层结构，用来快速搭出可运行的神经网络。',
    details: [
      '模板通常包括输入层、特征提取层和输出层，方便教学时快速切换实验场景。',
      '理解模板结构有助于把不同模式下的界面和底层网络一一对应起来。'
    ],
    mode: 'A'
  },
  {
    id: 'conv2d',
    title: '卷积层',
    aliases: ['conv', 'conv2d', '二维卷积'],
    category: '网络层',
    summary: '卷积层用一个小核在输入特征图上滑动，通过局部加权求和提取模式。',
    details: [
      '每个输出通道都对应一组卷积核权重，它会在输入的局部窗口上重复计算，生成新的特征图。',
      '卷积层特别适合捕捉边缘、纹理、局部形状和位置相关模式，因此是 CNN 的核心。'
    ],
    mode: 'A'
  },
  {
    id: 'kernel-size',
    title: '卷积核大小',
    aliases: ['kernel size', '核大小'],
    category: '层参数',
    summary: '卷积核大小决定了一次卷积会观察多大的局部区域。',
    details: [
      '3×3 核是最常见的配置，既能覆盖足够局部的结构，又不会让参数量过大。',
      '核越大，感受野越宽，但一次计算的成本也更高。'
    ],
    mode: 'A'
  },
  {
    id: 'stride',
    title: 'Stride',
    aliases: ['步幅', 'stride'],
    category: '层参数',
    summary: 'Stride 表示卷积核或池化窗口每次移动时跨过多少格。',
    details: [
      '步幅为 1 时会逐格滑动，输出保留更多细节；步幅增大则输出尺寸更小。',
      '步幅会直接影响输出 shape 和空间分辨率。'
    ],
    mode: 'A'
  },
  {
    id: 'padding',
    title: 'Padding',
    aliases: ['填充', 'padding'],
    category: '层参数',
    summary: 'Padding 会在输入边缘补值，用于控制边界信息保留和输出尺寸。',
    details: [
      '没有 padding 时，卷积窗口无法完整覆盖边缘，输出通常会变小。',
      '适当填充可以保留更多边界信息，也能让输出尺寸更接近输入尺寸。'
    ],
    mode: 'A'
  },
  {
    id: 'dilation',
    title: 'Dilation',
    aliases: ['空洞卷积', '膨胀卷积'],
    category: '层参数',
    summary: 'Dilation 通过拉开卷积核采样点间距，在不增加核尺寸的情况下扩大感受野。',
    details: [
      '普通卷积的 dilation 为 1，更大的 dilation 会让采样点以更稀疏的方式分布。',
      '它适合在保持参数量相对稳定的同时看见更大的上下文。'
    ],
    mode: 'A'
  },
  {
    id: 'bias',
    title: 'Bias',
    aliases: ['偏置', 'bias'],
    category: '层参数',
    summary: 'Bias 是在线性加权求和之后额外加上的可学习偏移量。',
    details: [
      '卷积层和全连接层都常在权重求和后加 bias，再送入后续激活函数。',
      'Bias 可以整体抬高或压低某个输出通道或某个类别的响应。'
    ],
    mode: 'A'
  },
  {
    id: 'activation',
    title: '激活函数',
    aliases: ['ReLU', 'Sigmoid', 'Tanh', '激活层'],
    category: '网络层',
    summary: '激活函数给网络引入非线性，使模型能够表达更复杂的模式。',
    details: [
      '如果只有线性层叠加，整个网络仍可以折叠成一个线性变换，表达能力会受限。',
      'ReLU 是最常见的激活之一，它会把负值截断到 0，保留正向响应。'
    ],
    mode: 'A'
  },
  {
    id: 'pooling',
    title: '池化层',
    aliases: ['池化', 'max pooling', '最大池化'],
    category: '网络层',
    summary: '池化层把局部区域压缩成更少的数值，用更低分辨率保留关键信息。',
    details: [
      '最大池化会从窗口中取最大值，强调最强响应；平均池化则更平滑。',
      '池化可以降低计算量，也能增强模型对微小平移的稳定性。'
    ],
    mode: 'A'
  },
  {
    id: 'flatten',
    title: 'Flatten',
    aliases: ['展平', '拉平'],
    category: '网络层',
    summary: 'Flatten 把多维特征图拉平成一维向量，作为全连接输出层的输入。',
    details: [
      '卷积和池化阶段保留了空间结构，而 flatten 会把这些结构摊平成一条特征向量。',
      '展平顺序必须和训练时一致，否则输出层接收到的特征位置就会错位。'
    ],
    mode: 'A'
  },
  {
    id: 'dense',
    title: 'Dense 全连接层',
    aliases: ['dense', '全连接层', '输出层'],
    category: '网络层',
    summary: 'Dense 层会把输入向量与权重矩阵相乘，并输出新的特征或分类分数。',
    details: [
      '每个输出神经元都连接到上一层全部输入，因此称为全连接层。',
      '在分类任务里，最后一个 dense 层通常直接产出各类别的原始分数。'
    ],
    mode: 'A'
  },
  {
    id: 'dropout',
    title: 'Dropout',
    aliases: ['随机失活', '随机丢弃'],
    category: '网络层',
    summary: 'Dropout 会在训练时随机屏蔽部分神经元，用来降低过拟合。',
    details: [
      '它更多属于训练阶段机制，推理阶段通常关闭以获得稳定输出。',
      '教学页面里如果展示 dropout，重点通常不是分类结果，而是帮助理解训练正则化思想。'
    ],
    mode: 'A'
  },
  {
    id: 'layer-inspector',
    title: '层检查器',
    aliases: ['层详情', '层解释面板', 'inspector'],
    category: '前向传播流程',
    summary: '层检查器会聚焦当前选中的层，展示它的输入、输出、参数和统计信息。',
    details: [
      '它相当于把拓扑图中的一个节点展开，帮助回答“这一层具体做了什么”。',
      'Mode C 的细节面板和中间过程视图，本质上都属于层检查器的一部分。'
    ],
    mode: 'A'
  },
  {
    id: 'feature-map',
    title: '特征图',
    aliases: ['feature map', '激活图', '响应图'],
    category: '输出解读',
    summary: '特征图是卷积层或激活层输出的空间响应图，用来表示模型在不同位置看到了什么。',
    details: [
      '一张特征图通常对应一个输出通道，亮度或颜色变化表示该位置响应强弱。',
      '比较不同通道的特征图，可以帮助理解模型在同一层里分别关注了哪些模式。'
    ],
    mode: 'A'
  },
  {
    id: 'tensor-stats',
    title: '张量统计',
    aliases: ['均值', '最小值', '最大值', '统计值'],
    category: '输出解读',
    summary: '张量统计用少量数字概括当前层输出的数值分布。',
    details: [
      '均值、最小值、最大值和能量可以帮助判断这一层是否几乎没响应，或是否响应过强。',
      '它们常用来配合特征图，避免只看颜色却忽略了真实数值。'
    ],
    mode: 'A'
  },
  {
    id: 'top-k',
    title: 'Top-K',
    aliases: ['前 K 类', '概率排名', '前几名类别'],
    category: '输出解读',
    summary: 'Top-K 会列出当前输出中分数最高的若干类别，帮助快速理解模型的判断倾向。',
    details: [
      'Top-1 是最可能的类别，而 Top-3 或 Top-5 更适合观察模型的犹豫和混淆方向。',
      '在 Mode C 中，完整的 softmax 排名会比只看第一名更有教学价值。'
    ],
    mode: 'A'
  }
];

export const MODE_C_TEACHING_TERMS: TeachingTerm[] = [
  {
    id: 'intermediate-view',
    title: '中间过程视图',
    aliases: ['Intermediate View', '逐步解释视图', '中间解释弹窗'],
    category: '前向传播流程',
    summary: '中间过程视图会把某一层的内部计算拆开，展示从输入局部到输出响应的完整链路。',
    details: [
      '在 Mode C 中，卷积、ReLU 和池化都不只是显示最终结果，还会用中间过程视图把局部计算展开。',
      '这类视图适合回答“这个输出值是怎么来的”，而不仅是“这一层长什么样”。',
      '因此它是 Mode C 相比普通 CNN 可视化更强调“教学解释”的关键部分。'
    ],
    mode: 'C'
  },
  {
    id: 'conv-patch',
    title: '卷积局部块',
    aliases: ['patch', '卷积 patch', '3×3 输入窗口'],
    category: '网络层',
    summary: '卷积局部块是卷积核当前覆盖到的那一小块输入区域，它只对应一个当前输出位置。',
    details: [
      '一次标准 3×3 卷积会从输入特征图里取一个 3×3 patch，再与卷积核逐元素相乘。',
      '把这个 patch 的乘积求和并加上 bias 后，只会得到当前输出通道上的一个输出格，而不是一整片区域。',
      'Mode C 会把这个局部块单独列出来，就是为了把“局部输入如何映射成单个输出值”讲清楚。'
    ],
    mode: 'C'
  },
  {
    id: 'pool-window',
    title: '池化窗口',
    aliases: ['pool window', '2×2 池化窗口', '局部比较窗口'],
    category: '网络层',
    summary: '池化窗口是一小块局部区域，池化层会在这里面做比较或聚合，然后输出一个值。',
    details: [
      '最大池化不会学习新的权重，而是直接从窗口中的若干候选值里选出最大的那个。',
      '对 2×2 最大池化来说，输入窗口里有 4 个数，但输出只有 1 个数，因此空间尺寸会被压缩。',
      'Mode C 会把这 4 个候选值和最终 max 结果同时列出来，让池化过程更直观。'
    ],
    mode: 'C'
  },
  {
    id: 'softmax-ranking',
    title: 'Softmax 排名',
    aliases: ['softmax ranking', '类别概率排序', '完整输出排名'],
    category: '输出解读',
    summary: 'Softmax 排名把输出层原始分数归一化成概率，并按从高到低排列，显示模型当前最倾向的类别。',
    details: [
      '排名第一的类别通常被当作当前预测，但第二名、第三名同样很有教学价值，因为它们反映了模型的混淆方向。',
      'Mode C 会展示完整的 10 类排名，而不是只截取前三类，这样更方便观察类别竞争关系。',
      '如果某个样本的第一名和第二名非常接近，往往意味着模型对它仍然存在不确定性。'
    ],
    mode: 'C'
  },
  {
    id: 'output-logit',
    title: '输出 Logit',
    aliases: ['logit', '未归一化分数', '原始类别分数'],
    category: '输出解读',
    summary: 'Logit 是 softmax 之前的原始类别分数，它决定了 softmax 排名的相对顺序。',
    details: [
      '输出层会把 flatten 后的特征向量与每个类别的权重做加权求和，再加上 bias，得到 logits。',
      '这些分数本身不是概率，也不要求落在 0 到 1 之间；只有 softmax 后才变成概率分布。',
      '理解 logit 能帮助你知道“为什么这个类别排第一”，而不是只看到最后的概率结果。'
    ],
    mode: 'C'
  },
  {
    id: 'channel-response',
    title: '通道响应',
    aliases: ['channel response', '通道激活', '通道关注区域'],
    category: '输出解读',
    summary: '通道响应描述某个输出通道在当前样本上对哪些区域反应更强，也就是特征图里哪些位置更突出。',
    details: [
      '同一层的不同通道通常学习到不同类型的模式，因此比较不同通道的响应可以帮助理解网络的分工。',
      '强响应并不一定代表预测正确，但它能说明模型在图像上把注意力放到了哪里。',
      'Mode C 把通道响应和中间过程视图放在一起，是为了同时解释“看到哪里”与“为什么这么看”。'
    ],
    mode: 'C'
  }
];

export const MODE_B_TEACHING_TERMS: TeachingTerm[] = [
  {
    id: 'training-dataset',
    title: '训练数据集',
    aliases: ['dataset', '训练数据', '带标签数据'],
    category: 'B 模式 · 数据与训练',
    summary: '训练数据集是模型学习规律所使用的样本集合，通常包含输入特征以及对应标签或目标值。',
    details: [
      '图像分类数据通常按类别组织图片，CSV 数据则需要明确特征列和标签列。',
      '输入形状、标签类型和类别数量必须与网络输入层、输出层及损失函数保持一致。',
      '数据质量、类别平衡和样本数量会直接影响训练曲线与最终泛化能力。'
    ],
    mode: 'B'
  },
  {
    id: 'dataset-split',
    title: '训练集、验证集与测试集',
    aliases: ['数据划分', 'train/val/test split', '训练验证测试划分'],
    category: 'B 模式 · 数据与训练',
    summary: '数据划分把样本分成训练、验证和测试三部分，分别用于学习参数、调整方案和最终评估。',
    details: [
      '训练集参与反向传播和参数更新；验证集不更新参数，用于观察泛化趋势和选择超参数。',
      '测试集应在训练方案确定后使用，避免模型或调参过程间接记住测试数据。',
      '常见比例是 70/15/15 或 80/10/10，样本较少时需要特别关注每个类别是否都被合理分配。'
    ],
    mode: 'B'
  },
  {
    id: 'hyperparameter',
    title: '训练超参数',
    aliases: ['hyperparameter', '训练配置', '调参'],
    category: 'B 模式 · 数据与训练',
    summary: '超参数是在训练开始前设定的配置，例如批大小、训练轮数、学习率、优化器和损失函数。',
    details: [
      '超参数不会像网络权重一样由反向传播直接学得，而是由实验者根据任务与训练表现调整。',
      '比较实验时应同时记录超参数、网络结构、数据划分和随机条件，否则指标差异很难解释。',
      '调参应一次改变少量关键变量，并保留 checkpoint 和曲线作为依据。'
    ],
    mode: 'B'
  },
  {
    id: 'epoch',
    title: 'Epoch',
    aliases: ['训练轮次', '训练轮数', '轮次'],
    category: 'B 模式 · 数据与训练',
    summary: '一个 Epoch 表示模型完整遍历一次训练集。',
    details: [
      '每个 Epoch 通常由多个 Batch 组成，训练集越大或批大小越小，每轮包含的 Batch 越多。',
      'Epoch 太少可能欠拟合，太多则可能让训练集继续变好但验证集开始变差。',
      '观察每轮结束后的训练与验证指标，可以判断是否需要继续训练或提前停止。'
    ],
    mode: 'B'
  },
  {
    id: 'batch',
    title: 'Batch',
    aliases: ['批次', '批大小', 'batch size'],
    category: 'B 模式 · 数据与训练',
    summary: 'Batch 是一次前向传播、反向传播和参数更新所处理的一小组样本。',
    details: [
      '批大小决定每次更新使用多少样本：较大批次更稳定但更占显存，较小批次噪声更大但更新更频繁。',
      '一个训练 step 通常对应一个 Batch 的处理和一次优化器更新。',
      '改变批大小时，合适的学习率也可能随之变化。'
    ],
    mode: 'B'
  },
  {
    id: 'optimizer',
    title: '优化器',
    aliases: ['optimizer', '参数优化算法'],
    category: 'B 模式 · 数据与训练',
    summary: '优化器根据梯度决定网络参数如何更新，常见选择包括 SGD、Momentum、Adam 和 AdamW。',
    details: [
      'SGD 更新规则直接、可控；Momentum 利用历史方向减小震荡；Adam 会为不同参数自适应调整更新尺度。',
      '优化器必须与学习率一起理解，相同学习率在不同优化器下可能产生完全不同的更新幅度。',
      '实验对比时应记录优化器及其附加参数，避免只比较最终准确率。'
    ],
    mode: 'B'
  },
  {
    id: 'learning-rate-scheduler',
    title: '学习率调度器',
    aliases: ['scheduler', '学习率调度', '学习率衰减'],
    category: 'B 模式 · 数据与训练',
    summary: '学习率调度器会在训练过程中按规则改变学习率，使模型前期快速学习、后期细致收敛。',
    details: [
      '阶梯衰减会在指定阶段降低学习率，余弦退火则让学习率沿平滑曲线逐渐变化。',
      '查看学习率曲线时，应把变化时刻与损失、准确率和梯度变化放在一起分析。',
      '衰减过早可能让模型停止有效学习，衰减太晚则可能在最优点附近持续震荡。'
    ],
    mode: 'B'
  },
  {
    id: 'loss-function',
    title: '损失函数与损失曲线',
    aliases: ['loss', '训练损失', '验证损失', 'loss curve'],
    category: 'B 模式 · 指标诊断',
    summary: '损失函数衡量预测与真实目标的差异，损失曲线展示这种误差随训练 step 或 Epoch 的变化。',
    details: [
      '分类常用交叉熵或二元交叉熵，回归常用均方误差；任务类型与输出层必须和损失函数匹配。',
      '训练损失下降说明模型正在拟合训练数据，验证损失则更能反映对未参与更新数据的泛化表现。',
      '若训练损失持续下降而验证损失上升，通常需要警惕过拟合。'
    ],
    mode: 'B'
  },
  {
    id: 'accuracy',
    title: '准确率',
    aliases: ['accuracy', '训练准确率', '验证准确率', '测试准确率'],
    category: 'B 模式 · 指标诊断',
    summary: '准确率表示预测正确的样本占全部样本的比例，是分类任务最直观的指标之一。',
    details: [
      '训练准确率用于观察拟合程度，验证准确率用于选择方案，测试准确率用于报告最终效果。',
      '类别极不平衡时，准确率可能掩盖少数类别表现，需要结合类别分布和其他指标分析。',
      '回归任务通常不使用分类准确率，而应关注 MSE、MAE 等误差指标。'
    ],
    mode: 'B'
  },
  {
    id: 'gradient-norm',
    title: '梯度范数',
    aliases: ['gradient norm', 'grad_norm', '梯度大小'],
    category: 'B 模式 · 指标诊断',
    summary: '梯度范数把一层或整个网络的梯度汇总成一个大小指标，用于观察反向信号是否稳定。',
    details: [
      '梯度长期接近零可能意味着梯度消失或层没有得到有效学习信号。',
      '梯度突然非常大可能导致参数剧烈变化、损失震荡或数值发散。',
      '判断是否异常要结合网络结构、激活函数、学习率和参数更新范数，而不能只看单个阈值。'
    ],
    mode: 'B'
  },
  {
    id: 'parameter-update',
    title: '参数更新',
    aliases: ['update norm', '权重更新', 'optimizer step'],
    category: 'B 模式 · 指标诊断',
    summary: '参数更新是优化器利用梯度实际改变权重和偏置的步骤。',
    details: [
      '梯度描述损失对参数的敏感程度，更新量则是学习率、优化器状态与梯度共同作用后的真实改变量。',
      '有梯度但更新量很小，可能与学习率过低或自适应优化器缩放有关。',
      '逐层比较梯度范数和 update norm，可以发现哪些层几乎没有学习或更新过猛。'
    ],
    mode: 'B'
  },
  {
    id: 'weight-distribution',
    title: '权重分布',
    aliases: ['weight histogram', '权重直方图', '参数分布'],
    category: 'B 模式 · 指标诊断',
    summary: '权重分布直方图展示模型参数落在不同数值区间的相对数量。',
    details: [
      '分布的中心、宽度和长尾会随着训练变化，可辅助判断参数是否集中、发散或出现异常极值。',
      '权重均值和标准差只能概括整体，直方图能显示多峰、偏斜等更细的结构。',
      '权重分布需要结合梯度、更新量和损失曲线共同解释。'
    ],
    mode: 'B'
  },
  {
    id: 'overfitting',
    title: '过拟合与欠拟合',
    aliases: ['overfitting', 'underfitting', '泛化差距'],
    category: 'B 模式 · 指标诊断',
    summary: '过拟合表示模型记住训练数据但泛化较差，欠拟合则表示模型连训练数据中的规律也未充分学会。',
    details: [
      '训练指标很好而验证指标明显较差，通常是过拟合信号；训练和验证指标都较差，则更像欠拟合。',
      '过拟合可尝试数据增强、正则化、Dropout、减小模型或提前停止；欠拟合可增加容量、训练轮数或调整优化配置。',
      '判断时应观察完整曲线和多次实验，避免根据单个 Epoch 的波动下结论。'
    ],
    mode: 'B'
  },
  {
    id: 'checkpoint',
    title: '模型 Checkpoint',
    aliases: ['checkpoint', '模型检查点', '.pt 权重'],
    category: 'B 模式 · 实验与推理',
    summary: 'Checkpoint 是训练过程中保存的模型状态，用于恢复训练、运行测试或复现实验结果。',
    details: [
      '平台中的 checkpoint 除权重文件外，还应关联网络结构、超参数、数据集、训练进度和指标。',
      '只有结构与权重匹配，模型才能正确加载；输入预处理和类别映射也应与训练时一致。',
      '实验对比应区分已完成、手动停止和异常结束的 checkpoint，避免把不完整结果直接横向比较。'
    ],
    mode: 'B'
  },
  {
    id: 'single-sample-inference',
    title: '单样本推理',
    aliases: ['single sample inference', '单条预测', '样本推理'],
    category: 'B 模式 · 实验与推理',
    summary: '单样本推理使用训练完成的 checkpoint 对一条数据执行前向传播，并展示预测与中间结果。',
    details: [
      '推理阶段不执行反向传播，也不会更新模型参数。',
      '样本预处理、特征顺序和类别映射必须与训练时一致，否则预测结果没有可比性。',
      '逐层查看激活可以帮助定位信息在哪一层被放大、压缩或丢失。'
    ],
    mode: 'B'
  },
  {
    id: 'layer-activation',
    title: '逐层激活',
    aliases: ['activation value', '中间激活', '层输出'],
    category: 'B 模式 · 实验与推理',
    summary: '逐层激活是单个样本经过网络各层后产生的中间张量或向量。',
    details: [
      '卷积层激活常以通道热力图展示，全连接层或表格模型激活常以向量柱状图展示。',
      '均值、最大值、最小值和非零比例可以概括当前层输出是否稀疏、饱和或异常。',
      '激活很强不等于该层一定对正确类别有正向贡献，需要结合最终预测和更多解释方法判断。'
    ],
    mode: 'B'
  },
  {
    id: 'prediction-confidence',
    title: '预测置信度',
    aliases: ['confidence', '预测概率', '类别概率'],
    category: 'B 模式 · 实验与推理',
    summary: '预测置信度通常是模型分配给预测类别的概率值，表示输出分布对该类别的集中程度。',
    details: [
      '置信度高只表示模型输出更集中，不保证预测一定正确，也不代表概率已经良好校准。',
      '当多个类别概率接近时，模型对当前样本更不确定，Top-K 分布比单个最大值更有信息。',
      '分析错误样本时，应同时查看真实标签、Top-K、输入数据和逐层激活。'
    ],
    mode: 'B'
  }
];

export const MODE_E_TEACHING_TERMS: TeachingTerm[] = [
  {
    id: 'optimizer-sgd',
    title: 'SGD 优化器',
    aliases: ['SGD', '随机梯度下降'],
    category: '训练与优化',
    summary: 'SGD 是最基础的优化器，会直接沿着负梯度方向更新参数。',
    details: [
      '它的更新规则简单直接，适合教学中展示”梯度如何驱动参数变化”的基本过程。',
      '缺点是对学习率更敏感，在复杂损失面上也可能收敛较慢。'
    ],
    mode: 'E'
  },
  {
    id: 'optimizer-momentum',
    title: 'Momentum 优化器',
    aliases: ['Momentum', '动量优化'],
    category: '训练与优化',
    summary: 'Momentum 在 SGD 的基础上累积历史更新方向，用惯性帮助参数更稳定地前进。',
    details: [
      '它能减轻来回震荡的问题，并在较平坦区域保持前进速度。',
      '教学上常把它类比成滚动的小球，帮助理解”为什么它比纯 SGD 更稳”。'
    ],
    mode: 'E'
  },
  {
    id: 'optimizer-adam',
    title: 'Adam 优化器',
    aliases: ['Adam', '自适应优化器'],
    category: '训练与优化',
    summary: 'Adam 结合了动量和自适应学习率，是深度学习中最常见的优化器之一。',
    details: [
      '它会根据一阶和二阶统计量自动调节参数更新尺度，因此通常收敛更快。',
      '教学上适合和 SGD、Momentum 做对比，观察不同优化器对训练曲线的影响。'
    ],
    mode: 'E'
  },
  {
    id: 'backpropagation',
    title: '反向传播',
    aliases: ['backpropagation', '梯度回传', '反向计算'],
    category: '训练与优化',
    summary: '反向传播利用链式法则把输出误差逐层回传，得到每层参数的梯度。',
    details: [
      '它是神经网络训练的核心机制，因为只有先得到梯度，优化器才能更新参数。',
      '前向传播负责算结果，反向传播负责告诉模型”哪里错了以及该怎么改”。'
    ],
    mode: 'E'
  },
  {
    id: 'gradient-descent',
    title: '梯度下降',
    aliases: ['gradient descent', '梯度下降法'],
    category: '训练与优化',
    summary: '梯度下降通过反复沿负梯度方向更新参数，使损失函数逐步减小。',
    details: [
      '它是大多数神经网络训练算法背后的基本思想，区别主要在于更新步子的设计方式。',
      '教学页面常用它来解释为什么多次迭代后模型会逐渐学会分类边界。'
    ],
    mode: 'E'
  },
  {
    id: 'learning-rate',
    title: '学习率',
    aliases: ['learning rate', 'lr'],
    category: '训练与优化',
    summary: '学习率控制每次参数更新走多大一步，是训练中最重要的超参数之一。',
    details: [
      '学习率太大可能导致训练震荡甚至发散，太小则会让训练极慢。',
      '观察训练曲线与决策边界变化，通常能直观看到学习率设置是否合适。'
    ],
    mode: 'E'
  },
  {
    id: 'activation-relu',
    title: 'ReLU',
    aliases: ['ReLU', '修正线性单元'],
    category: '训练与优化',
    summary: 'ReLU 输出 max(0, x)，反向传播时正半区梯度为 1，负半区为 0。',
    details: [
      '公式: f(x) = max(0, x)，导数: 正半区为 1、负半区为 0。',
      '优点: 计算简单、梯度不衰减，深层网络也能有效训练。缺点: 负半区神经元可能永久死亡。',
      '决策边界: 产生分段线性边界（多个半平面拼成），XOR 等直线分割用 ReLU 效果好，环形需要大量神经元。'
    ],
    mode: 'E'
  },
  {
    id: 'activation-sigmoid',
    title: 'Sigmoid',
    aliases: ['Sigmoid', 'Logistic'],
    category: '训练与优化',
    summary: 'Sigmoid 输出 (0,1)，导数最大 0.25，深层网络易梯度消失。',
    details: [
      '公式: f(x) = 1/(1+exp(-x))，导数: f(x)*(1-f(x)) <= 0.25。',
      '优点: 输出平滑有界。缺点: 深层网络梯度指数衰减、非零中心收敛慢。',
      '决策边界: 产生光滑弧形，适合同心圆等曲线边界。D 模式中同心圆用 Sigmoid 优于 ReLU。'
    ],
    mode: 'E'
  },
  {
    id: 'activation-tanh',
    title: 'Tanh',
    aliases: ['Tanh', '双曲正切'],
    category: '训练与优化',
    summary: 'Tanh 输出 (-1,1)，零中心化。梯度最大为 1，饱和时趋近 0。',
    details: [
      '公式: f(x) = tanh(x)，导数: 1 - tanh^2(x)，x=0 时最大为 1。',
      '优点: 零中心化、比 Sigmoid 梯度大。缺点: 远离原点时饱和导致梯度消失，多层叠加尤其严重。',
      '决策边界: 也产生光滑弧形，但深层网络梯度衰减比 ReLU 严重得多。'
    ],
    mode: 'E'
  }
];

export const MODE_F_TEACHING_TERMS: TeachingTerm[] = [
  {
    id: 'rnn-cell',
    title: 'RNN 单元',
    aliases: ['rnn cell', '循环单元', 'RNN 细胞'],
    category: '序列模型',
    summary: 'RNN 单元是循环网络的基本计算单元，在每一时间步接收输入和上一隐状态，产生新的隐状态和输出。',
    details: [
      '每个 RNN 单元的核心操作是: h_t = tanh(W_xh * x_t + W_hh * h_{t-1} + b_h)。',
      '隐状态 h_t 是 RNN 的"记忆"，它在时间步之间传递，捕捉序列中的模式。',
      '在 Mode F 中，可以直观看到每个时间步中隐状态向量的数值变化。'
    ],
    mode: 'F'
  },
  {
    id: 'bptt',
    title: 'BPTT (穿越时间反向传播)',
    aliases: ['BPTT', 'Backpropagation Through Time', '时间反向传播'],
    category: '序列模型',
    summary: 'BPTT 是 RNN 训练的核心算法，将网络沿时间展开后，用链式法则从最后一个时间步反向传播梯度到第一个时间步。',
    details: [
      'BPTT 把 RNN 沿时间维度展开成一个深层前馈网络，每个时间步对应一层。',
      '梯度从 t = T 开始回传，经过每个时间步的 tanh 导数，容易产生梯度消失或爆炸。',
      'Mode F 可视化了梯度的流动过程，可以观察梯度范数随训练如何变化。'
    ],
    mode: 'F'
  },
  {
    id: 'hidden-state',
    title: '隐状态 (Hidden State)',
    aliases: ['hidden state', '隐层状态', 'RNN 记忆'],
    category: '序列模型',
    summary: '隐状态是 RNN 在时间步之间传递信息的向量，承载了网络对前序输入的记忆和摘要。',
    details: [
      '隐状态的维度是网络的"记忆容量"，更宽的隐层可以存储更丰富的历史信息。',
      '每个时间步，隐状态会根据当前输入和上一状态进行更新: 新的输入加入，旧的部分信息通过非线性变换保留。',
      '在 Mode F 中，隐状态向量以柱状图形式在每个时间步展示，可以观察信息如何在时间步间演化。'
    ],
    mode: 'F'
  },
  {
    id: 'gradient-vanishing',
    title: '梯度消失',
    aliases: ['vanishing gradient', '梯度衰减', '梯度消失问题'],
    category: '序列模型',
    summary: '在 BPTT 中，当梯度沿时间反向传播时，经过多层 tanh 导数连乘，梯度迅速衰减到接近零。',
    details: [
      'tanh 的导数最大为 1，在饱和区接近 0，多个小于 1 的数连乘导致指数级衰减。',
      '梯度消失使得 RNN 难以学习长距离依赖——早期时间步的信息几乎无法影响权重更新。',
      '这是 LSTM 和 GRU 等改进结构诞生的主要原因，它们用门控机制缓解了这个问题。'
    ],
    mode: 'F'
  },
  {
    id: 'optimizer-sgd',
    title: 'SGD 优化器',
    aliases: ['SGD', '随机梯度下降'],
    category: '序列模型',
    summary: 'SGD 是最基础的优化器，直接沿负梯度方向更新参数。',
    details: [
      '在 RNN 训练中，SGD 对学习率更敏感，可能因梯度范数波动而震荡。',
      '对比 SGD、Momentum、Adam 在 RNN 任务上的表现，可以直观理解优化器的差异。'
    ],
    mode: 'F'
  },
  {
    id: 'optimizer-momentum',
    title: 'Momentum 优化器',
    aliases: ['Momentum', '动量优化'],
    category: '序列模型',
    summary: 'Momentum 在 SGD 基础上累积历史更新方向，帮助参数在梯度震荡的维度上更稳定地前进。',
    details: [
      '在 BPTT 中梯度在各时间步之间可能波动较大，Momentum 的惯性有助于平滑更新。',
      '可以对比观察 Momentum 和纯 SGD 在相同 RNN 任务上的损失曲线差异。'
    ],
    mode: 'F'
  },
  {
    id: 'optimizer-adam',
    title: 'Adam 优化器',
    aliases: ['Adam', '自适应优化器'],
    category: '序列模型',
    summary: 'Adam 结合动量和自适应学习率，对每个参数维护一阶和二阶矩估计，通常收敛更快更稳。',
    details: [
      'Adam 对 RNN 中可能出现的梯度范数波动有天然的适应能力，自动调节每个参数的步长。',
      '在 Mode F 中，可以切换 Adam 观察训练曲线和最终准确率的变化。'
    ],
    mode: 'F'
  },
  {
    id: 'sequence-classification',
    title: '序列分类',
    aliases: ['sequence classification', '时序分类'],
    category: '序列模型',
    summary: '序列分类任务要求模型在看完整个输入序列后，判断序列属于哪一类别。',
    details: [
      '在 Mode F 中，RNN 在最后一个时间步输出类别概率，用 softmax + 交叉熵计算损失。',
      '例如"延迟记忆"任务需要模型记住第 0 步看到的 bit，经过几步延迟后正确分类。',
      '这考验 RNN 的长期记忆能力——早期的关键信息能否被隐状态保留到最后。'
    ],
    mode: 'F'
  },
];

export const MODE_D_TEACHING_TERMS: TeachingTerm[] = [
  {
    id: 'transformer-token',
    title: 'Token',
    aliases: ['token', '词元', '子词'],
    category: 'Transformer 基础',
    summary: 'Token 是 Transformer 处理文本时使用的最小离散单位，一句话会先被切分成若干 token，再送入模型。',
    details: [
      '一个 token 不一定等于一个完整单词，它也可能只是单词的一部分、一个标点，或者一个常见短语。',
      '模式 D 中看到的 token 序列，就是模型当前正在计算注意力和下一词预测的输入单位。',
      '理解 token 很重要，因为 Top-K 预测、注意力矩阵和 QKV 计算都不是直接对“整句”发生，而是围绕这些离散 token 展开。'
    ],
    mode: 'D'
  },
  {
    id: 'next-token-prediction',
    title: '下一词预测',
    aliases: ['next token prediction', 'next word prediction', '自回归预测'],
    category: 'Transformer 基础',
    summary: '下一词预测指模型根据前文 token 序列，预测下一个最可能出现的 token。',
    details: [
      '模式 D 中的 Top-K 面板本质上是在展示“下一个 token 可能是谁”，而不是对整句话做分类。',
      '模型会先生成一组 logits，再通过 softmax 变成概率分布，最终选出概率最高的候选 token。',
      '如果前几名概率很接近，通常说明模型对后续词仍有不确定性；如果第一名显著领先，则说明上下文对预测约束更强。'
    ],
    mode: 'D'
  },
  {
    id: 'attention-matrix',
    title: '注意力矩阵',
    aliases: ['attention matrix', 'self-attention matrix', '注意力热力图'],
    category: '注意力机制',
    summary: '注意力矩阵描述序列中每个 token 在当前层当前头里，对其他 token 分配了多少关注权重。',
    details: [
      '矩阵中的一行通常表示“某个查询 token 正在看谁”，一列表示“某个 token 被多少查询 token 关注”。',
      '某个格子越大，表示对应的 Query 更依赖那个 Key 提供的信息。',
      '模式 D 的注意力可视化可以帮助解释为什么某个 token 预测会受到前文某些位置的强烈影响。'
    ],
    mode: 'D'
  },
  {
    id: 'attention-head',
    title: '注意力头',
    aliases: ['attention head', 'multi-head attention', 'head'],
    category: '注意力机制',
    summary: '注意力头是多头注意力中的一个独立子空间，每个头会学习自己偏好的关注模式。',
    details: [
      '不同头可能分别关注句法关系、位置邻近、重复词、实体指代或局部搭配。',
      '模式 D 里切换 head，本质上是在观察“同一层里，不同子空间各自看到了什么关系”。',
      '因此头与头之间不一定谁更“正确”，而是它们分工不同，共同形成最终输出。'
    ],
    mode: 'D'
  },
  {
    id: 'query-key-value',
    title: 'QKV',
    aliases: ['QKV', 'query key value', 'query', 'key', 'value'],
    category: '注意力机制',
    summary: 'QKV 是自注意力的核心三元组：Query 决定“我要找什么”，Key 决定“我能提供什么匹配线索”，Value 决定“真正传过去的信息内容”。',
    details: [
      '先用 Query 和所有 Key 做匹配，得到一组相关性分数；再把这些分数归一化成注意力权重；最后用这些权重对所有 Value 做加权求和。',
      '所以注意力强并不等于直接复制某个 token，而是说明它对应的 Value 更大概率被纳入当前输出。',
      '模式 D 的 QKV 面板就是把这个链条拆开，让你看到“为什么这一格注意力会高”以及“高了以后信息怎样流向输出”。'
    ],
    mode: 'D'
  },
  {
    id: 'scaled-dot-product',
    title: '缩放点积注意力',
    aliases: ['scaled dot-product attention', 'dot product attention', '缩放注意力'],
    category: '注意力机制',
    summary: '缩放点积注意力先计算 Query 与 Key 的点积相似度，再除以根号下的维度大小，最后经过 softmax 得到权重。',
    details: [
      '点积越大，说明当前 Query 和某个 Key 越匹配，对应格子的注意力分数通常也越高。',
      '除以根号维度是为了避免维度较大时分数过大，导致 softmax 过于尖锐、训练或解释都不稳定。',
      '模式 D 中从 Q×K 到缩放再到 softmax 的步骤，正是这一计算链条的教学化展开。'
    ],
    mode: 'D'
  },
  {
    id: 'softmax-attention',
    title: '注意力 Softmax',
    aliases: ['attention softmax', 'attention weights', '注意力权重'],
    category: '注意力机制',
    summary: '注意力 softmax 会把原始相关性分数转换成和为 1 的权重分布，表示当前 Query 如何在各个 Key 之间分配注意力。',
    details: [
      'softmax 后的结果更容易解释，因为每个数都可以看成“当前关注预算里分给某位置的比例”。',
      '如果某一行特别尖锐，说明当前 Query 主要盯着少数几个 token；如果比较平缓，则说明它在综合更广的上下文。',
      '模式 D 的注意力矩阵和 QKV 面板都可以把 softmax 之后的权重视作“信息流量分配图”。'
    ],
    mode: 'D'
  },
  {
    id: 'context-vector',
    title: '上下文向量',
    aliases: ['context vector', 'attention output', '注意力输出'],
    category: '注意力机制',
    summary: '上下文向量是注意力权重对所有 Value 加权求和后的结果，它代表当前 token 在这一头里汇总得到的信息。',
    details: [
      '这个向量不是单个 Value 的拷贝，而是多个 Value 按权重混合后的结果。',
      '它反映了“当前 Query 在这一头里最终带走了什么信息”，会继续流向后续线性层、残差连接和下一层。',
      '模式 D 里 Value 汇入输出的可视化，就是在帮助理解上下文向量是怎样被组装出来的。'
    ],
    mode: 'D'
  },
  {
    id: 'top-k-token',
    title: 'Top-K 候选词',
    aliases: ['top-k token', '候选词', '下一词候选'],
    category: '输出解读',
    summary: 'Top-K 候选词展示当前下一词预测中概率最高的若干 token，用来快速理解模型当前的语言倾向。',
    details: [
      '第一名是当前最可能输出的 token，但第二名、第三名也很有教学价值，因为它们能反映模型的犹豫方向。',
      '如果多个候选词语义接近，通常说明模型已经抓住了大致语境；如果候选词差异很大，则说明上下文约束较弱。',
      '模式 D 中把 Top-K 和注意力矩阵对照起来看，往往能更好解释“为什么最后会是这个词”。'
    ],
    mode: 'D'
  },
  {
    id: 'residual-connection',
    title: '残差连接',
    aliases: ['residual connection', 'skip connection', '残差'],
    category: 'Transformer 结构',
    summary: '残差连接会把子层的输入直接加回输出，帮助信息保留和深层训练稳定。',
    details: [
      '在 Transformer 中，注意力输出和前馈网络输出通常都不是孤立使用，而是要和原输入通过残差相加。',
      '这样做可以避免信息在多层传递中被完全覆盖，也能缓解深层网络优化困难的问题。',
      '即使模式 D 当前重点不在完整残差展开，理解它也有助于知道“这一层的输出并不是凭空生成，而是在原表示上逐步修正”。'
    ],
    mode: 'D'
  },
  {
    id: 'layernorm',
    title: 'LayerNorm',
    aliases: ['layer norm', 'layer normalization', '层归一化'],
    category: 'Transformer 结构',
    summary: 'LayerNorm 会在特征维度上对表示做归一化，让不同 token 的数值分布更稳定。',
    details: [
      '它不会改变 token 序列长度，但会调整每个位置向量内部的尺度和偏移。',
      '在 Transformer 中，LayerNorm 常与残差连接配合使用，帮助训练更稳定、数值更可控。',
      '从教学角度看，它可以理解为“在进入下一步计算前，先把当前表示重新整理到更容易处理的范围内”。'
    ],
    mode: 'D'
  }
];

export const TEACHING_TERMS = [
  ...MODE_A_TEACHING_TERMS,
  ...MODE_B_TEACHING_TERMS,
  ...MODE_C_TEACHING_TERMS,
  ...MODE_D_TEACHING_TERMS,
  ...MODE_E_TEACHING_TERMS,
  ...MODE_F_TEACHING_TERMS
];

/** 按术语 id 查找教学条目，供界面把“卷积”“softmax”等概念链接到对应解释。 */
export function findTeachingTerm(id: string): TeachingTerm | undefined {
  return TEACHING_TERMS.find(term => term.id === id);
}
