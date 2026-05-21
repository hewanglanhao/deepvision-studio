import { CommonModule } from '@angular/common';
import { AfterViewInit, Component, ElementRef, OnDestroy, ViewChild } from '@angular/core';
import { RouterLink } from '@angular/router';
import * as THREE from 'three';
import { PointerLockControls } from 'three/examples/jsm/controls/PointerLockControls.js';

type ExhibitKind =
  | 'logic'
  | 'perceptron'
  | 'eliza'
  | 'expert'
  | 'hopfield'
  | 'qlearning'
  | 'svm'
  | 'winter'
  | 'backprop'
  | 'cnn'
  | 'rnn'
  | 'deep'
  | 'yolo'
  | 'rl'
  | 'transformer'
  | 'llm'
  | 'alignment'
  | 'grpo'
  | 'multimodal';

interface MuseumExhibit {
  year: string;
  title: string;
  subtitle: string;
  description: string;
  bullets: string[];
  tags: string[];
  kind: ExhibitKind;
  position: THREE.Vector3;
  rotationY: number;
  accent: string;
  source: string;
}

interface MovingState {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
  sprint: boolean;
}

const HALL_WIDTH = 18;
const HALL_LENGTH = 164;
const WALL_HEIGHT = 7.2;
const CAMERA_HEIGHT = 1.75;

@Component({
  selector: 'app-ai-museum-page',
  standalone: true,
  imports: [CommonModule, RouterLink],
  templateUrl: './ai-museum-page.component.html',
  styleUrl: './ai-museum-page.component.css'
})
export class AiMuseumPageComponent implements AfterViewInit, OnDestroy {
  @ViewChild('stage') private stageRef?: ElementRef<HTMLDivElement>;

  readonly exhibits: MuseumExhibit[] = [
    {
      year: '1943',
      title: '人工神经元的数学雏形',
      subtitle: 'McCulloch 与 Pitts 用逻辑门描述神经活动',
      description: '早期 AI 的起点不是大模型，而是一个非常朴素的问题：能否把神经元看成可计算的逻辑单元？',
      bullets: ['形式化：y = 1[Σwᵢxᵢ ≥ θ]', '神经元被抽象成输入、阈值和输出', '符号逻辑与计算模型开始汇合'],
      tags: ['逻辑', '神经元', '连接主义'],
      kind: 'logic',
      position: new THREE.Vector3(-8.72, 2.55, -62),
      rotationY: Math.PI / 2,
      accent: '#38bdf8',
      source: 'McCulloch-Pitts neuron, 1943'
    },
    {
      year: '1950',
      title: '图灵提出机器智能问题',
      subtitle: '从“机器会思考吗”转向可观察的行为测试',
      description: '图灵把抽象哲学问题转化成实验问题：如果机器在对话中表现得像人，我们该如何评价它？',
      bullets: ['Imitation Game 把智能变成可测试行为', '自然语言成为 AI 的核心入口', '对话能力贯穿后来的大模型发展'],
      tags: ['Turing Test', '对话', '智能定义'],
      kind: 'logic',
      position: new THREE.Vector3(8.72, 2.55, -55),
      rotationY: -Math.PI / 2,
      accent: '#22c55e',
      source: 'Alan Turing, Computing Machinery and Intelligence, 1950'
    },
    {
      year: '1956',
      title: 'Dartmouth：AI 正式命名',
      subtitle: '人工智能成为一个研究领域',
      description: 'Dartmouth 研讨会让“Artificial Intelligence”成为共同旗帜，符号推理、搜索、规划等方向快速发展。',
      bullets: ['AI 作为研究领域被组织起来', '早期系统强调规则、搜索和推理', '乐观预期也埋下第一次 AI 寒冬的伏笔'],
      tags: ['Dartmouth', '符号主义', '搜索'],
      kind: 'logic',
      position: new THREE.Vector3(-8.72, 2.55, -48),
      rotationY: Math.PI / 2,
      accent: '#f59e0b',
      source: 'Dartmouth workshop proposal, 1956'
    },
    {
      year: '1957-1958',
      title: '感知机：会学习的机器',
      subtitle: 'Rosenblatt 的 Mark I Perceptron 让机器学习走上展台',
      description: '感知机把权重更新、分类边界和硬件实现联系在一起，是早期机器学习最具象的展品。',
      bullets: ['判别函数：ŷ = sign(w·x + b)', '错误样本触发更新：w ← w + η(y - ŷ)x', '只能表示线性可分边界，XOR 暴露局限'],
      tags: ['Perceptron', '监督学习', '权重更新'],
      kind: 'perceptron',
      position: new THREE.Vector3(8.72, 2.55, -41),
      rotationY: -Math.PI / 2,
      accent: '#fb7185',
      source: 'Frank Rosenblatt / Mark I Perceptron'
    },
    {
      year: '1966',
      title: 'ELIZA：早期自然语言交互',
      subtitle: '模式匹配也能制造“会聊天”的错觉',
      description: 'ELIZA 并不理解语义，它靠模板和关键词重写输入，却让人第一次直观看到机器对话界面的力量。',
      bullets: ['规则模板：关键词 → 回复脚本', '暴露“表面语言行为”和“真正理解”的差距', '从这里能一路看到 Chatbot、RLHF 与 LLM 助手'],
      tags: ['NLP', 'Chatbot', 'Pattern Matching'],
      kind: 'eliza',
      position: new THREE.Vector3(-8.72, 2.55, -34),
      rotationY: Math.PI / 2,
      accent: '#f472b6',
      source: 'Joseph Weizenbaum, ELIZA, 1966'
    },
    {
      year: '1970s-1980s',
      title: '专家系统：把知识写成规则',
      subtitle: 'IF-THEN 规则、知识库和推理机撑起第二波 AI 热潮',
      description: '专家系统证明 AI 可以在窄领域做出有价值的判断，但知识获取、规则维护和泛化能力成为瓶颈。',
      bullets: ['知识库：IF 症状 THEN 诊断/建议', '推理机负责规则匹配和冲突消解', '维护成本和脆弱性推动后来统计学习兴起'],
      tags: ['Expert System', 'Knowledge Base', 'Rules'],
      kind: 'expert',
      position: new THREE.Vector3(8.72, 2.55, -27),
      rotationY: -Math.PI / 2,
      accent: '#facc15',
      source: 'MYCIN, XCON and expert-system era'
    },
    {
      year: '1969-1980s',
      title: 'AI 寒冬与反思',
      subtitle: '能力边界被看见，研究转向更扎实的表示和学习',
      description: '早期系统的局限、计算资源不足和过高预期导致热潮降温。这里不是空白期，而是规则系统、统计学习和连接主义重新分化的过渡区。',
      bullets: ['感知机无法解决 XOR 等非线性问题', '专家系统维护成本高', '低谷推动研究者重新思考学习算法'],
      tags: ['AI Winter', 'XOR', '专家系统'],
      kind: 'winter',
      position: new THREE.Vector3(-8.72, 2.55, -20),
      rotationY: Math.PI / 2,
      accent: '#94a3b8',
      source: 'Minsky & Papert, Perceptrons; AI winter histories'
    },
    {
      year: '1982',
      title: 'Hopfield 网络：能量函数与联想记忆',
      subtitle: '神经网络被看成会收敛到稳定状态的动力系统',
      description: 'Hopfield 网络把神经元状态、权重对称性和能量最小化联系起来，是连接主义复兴中的关键节点。',
      bullets: ['状态更新会降低能量 E', '网络可作为联想记忆恢复受损模式', '能量模型思想后来影响 Boltzmann Machine 等方向'],
      tags: ['Hopfield', 'Energy Model', 'Memory'],
      kind: 'hopfield',
      position: new THREE.Vector3(8.72, 2.55, -13),
      rotationY: -Math.PI / 2,
      accent: '#c084fc',
      source: 'John Hopfield, Neural networks and physical systems, 1982'
    },
    {
      year: '1986',
      title: '反向传播：误差穿过多层网络',
      subtitle: '链式法则成为深度网络训练的发动机',
      description: '反向传播把 loss 对每一层参数的影响拆成局部梯度相乘，让多层网络从“能表达”变成“可训练”。',
      bullets: ['前向计算得到 loss', '反向传播 δ：从输出层逐层回传', '参数更新：θ ← θ - η∂L/∂θ'],
      tags: ['Backprop', 'Chain Rule', 'Gradient Descent'],
      kind: 'backprop',
      position: new THREE.Vector3(-8.72, 2.55, -6),
      rotationY: Math.PI / 2,
      accent: '#a78bfa',
      source: 'Rumelhart, Hinton & Williams, 1986'
    },
    {
      year: '1989',
      title: 'Q-learning：无模型强化学习',
      subtitle: '从试错中学习动作价值函数',
      description: 'Q-learning 用 Bellman 更新估计 Q(s,a)，为后来 DQN 把深度网络接入强化学习打下基础。',
      bullets: ['更新：Q ← Q + α[r + γmaxQ(s′,a′) - Q]', '无需环境转移模型', '价值函数思想贯穿 DQN 和后续 RL 系统'],
      tags: ['Q-learning', 'Bellman', 'RL'],
      kind: 'qlearning',
      position: new THREE.Vector3(8.72, 2.55, 1),
      rotationY: -Math.PI / 2,
      accent: '#84cc16',
      source: 'Watkins, Learning from Delayed Rewards, 1989'
    },
    {
      year: '1995',
      title: '支持向量机：最大间隔分类器',
      subtitle: 'Cortes 与 Vapnik 将统计学习理论推向实用高峰',
      description: 'SVM 的核心不是堆网络层，而是在特征空间里寻找最大间隔超平面；核技巧让线性模型进入高维非线性空间。',
      bullets: ['目标：最大化 margin = 2 / ||w||', '只有支持向量决定分界面', '核函数 K(xᵢ,xⱼ) 避免显式映射高维特征'],
      tags: ['SVM', 'Margin', 'Kernel Trick'],
      kind: 'svm',
      position: new THREE.Vector3(-8.72, 2.55, 8),
      rotationY: Math.PI / 2,
      accent: '#0ea5e9',
      source: 'Cortes & Vapnik, Support-Vector Networks, 1995'
    },
    {
      year: '1998',
      title: 'LeNet-5：CNN 进入视觉识别',
      subtitle: '局部感受野、权重共享、池化形成视觉网络模板',
      description: 'LeNet 展示了端到端训练的卷积网络如何读取手写数字，也奠定了后续视觉模型的基本模块。',
      bullets: ['卷积核扫描局部区域，提取边缘和纹理', '池化降低空间尺寸并增强平移鲁棒性', '全连接层将特征组合成分类决策'],
      tags: ['CNN', 'LeNet-5', 'Feature Map'],
      kind: 'cnn',
      position: new THREE.Vector3(8.72, 2.55, 22),
      rotationY: -Math.PI / 2,
      accent: '#06b6d4',
      source: 'LeCun, Bottou, Bengio & Haffner, 1998'
    },
    {
      year: '1997-2006',
      title: 'RNN 与 LSTM：记住时间序列',
      subtitle: '从循环状态到门控记忆，序列建模开始成熟',
      description: 'RNN 将隐藏状态沿时间传递，LSTM 用输入门、遗忘门、输出门缓解长程依赖中的梯度消失问题。',
      bullets: ['RNN：hₜ = f(Wxₜ + Uhₜ₋₁)', 'LSTM 用 cell state 保存长期信息', '语言建模、语音识别、机器翻译长期依赖它'],
      tags: ['RNN', 'LSTM', 'Sequence'],
      kind: 'rnn',
      position: new THREE.Vector3(8.72, 2.55, 15),
      rotationY: -Math.PI / 2,
      accent: '#ec4899',
      source: 'Hochreiter & Schmidhuber, Long Short-Term Memory, 1997'
    },
    {
      year: '2012',
      title: '深度学习突破视觉识别',
      subtitle: 'AlexNet 在 ImageNet 上展示深层 CNN 的威力',
      description: 'GPU、数据集和深层网络在这一刻汇合，深度学习从研究方向变成工程主线。',
      bullets: ['ReLU 缓解深层网络训练困难', 'Dropout 和数据增强降低过拟合', 'GPU 并行卷积改变训练规模'],
      tags: ['ImageNet', 'AlexNet', 'GPU'],
      kind: 'deep',
      position: new THREE.Vector3(-8.72, 2.55, 29),
      rotationY: Math.PI / 2,
      accent: '#06b6d4',
      source: 'AlexNet / ImageNet 2012'
    },
    {
      year: '2015',
      title: 'YOLO：一次前向传播完成检测',
      subtitle: '把目标检测改写成统一的实时回归问题',
      description: 'YOLO 不再先提候选框再分类，而是把图像划成网格，一次网络输出类别、置信度和边界框。',
      bullets: ['S×S 网格负责预测中心落入该格的目标', '每格输出 bounding boxes + confidence + class', '速度优势让实时视觉系统成为可能'],
      tags: ['Object Detection', 'YOLO', 'Real-time'],
      kind: 'yolo',
      position: new THREE.Vector3(8.72, 2.55, 36),
      rotationY: -Math.PI / 2,
      accent: '#ef4444',
      source: 'Redmon et al., You Only Look Once, 2015'
    },
    {
      year: '2015-2017',
      title: '深度强化学习：从 DQN 到 PPO',
      subtitle: '智能体在环境中试错，策略优化变成深度学习问题',
      description: 'DQN 用深度网络从像素预测动作价值；PPO 用裁剪目标稳定策略更新，成为许多 RLHF 系统的基础优化器。',
      bullets: ['DQN：Q(s,a) 估计动作长期回报', 'Experience Replay 打破样本相关性', 'PPO：clip ratio 限制策略更新幅度'],
      tags: ['DQN', 'PPO', 'Policy Optimization'],
      kind: 'rl',
      position: new THREE.Vector3(-8.72, 2.55, 43),
      rotationY: Math.PI / 2,
      accent: '#84cc16',
      source: 'Mnih et al. 2015; Schulman et al. PPO 2017'
    },
    {
      year: '2017',
      title: 'Transformer：注意力成为主角',
      subtitle: 'Attention Is All You Need',
      description: 'Transformer 去掉循环结构，用自注意力让每个 token 直接关注其他 token，成为现代大模型的核心架构。',
      bullets: ['Attention(Q,K,V)=softmax(QKᵀ/√d)V', '多头注意力并行捕捉语义、位置、指代关系', '残差、LayerNorm、FFN 组成可堆叠模块'],
      tags: ['Transformer', 'Self-Attention', 'Q/K/V'],
      kind: 'transformer',
      position: new THREE.Vector3(8.72, 2.55, 50),
      rotationY: -Math.PI / 2,
      accent: '#f97316',
      source: 'Vaswani et al., Attention Is All You Need, 2017'
    },
    {
      year: '2018-2023',
      title: '预训练大模型与生成式 AI',
      subtitle: '从 BERT/GPT 到 ChatGPT 与多模态模型',
      description: '大规模预训练把语言、代码、图像和工具调用连接起来，AI 从专用模型走向通用助手。',
      bullets: ['自监督预训练把海量文本转成 token 预测任务', '指令微调让模型学会按人类任务格式回答', '上下文学习把 prompt 变成临时程序'],
      tags: ['LLM', 'GPT', '多模态'],
      kind: 'llm',
      position: new THREE.Vector3(-8.72, 2.55, 57),
      rotationY: Math.PI / 2,
      accent: '#34d399',
      source: 'BERT, GPT series, ChatGPT, multimodal foundation models'
    },
    {
      year: '2022',
      title: 'RLHF：让模型更贴近人类偏好',
      subtitle: '从监督微调、奖励模型到 PPO 对齐',
      description: 'RLHF 把人类比较偏好转成奖励模型，再用强化学习优化语言模型输出，是 InstructGPT/ChatGPT 类助手的重要训练阶段。',
      bullets: ['SFT：用人工示范建立基础指令能力', 'Reward Model：学习“哪个回答更好”', 'PPO：在不偏离原模型太远的前提下最大化奖励'],
      tags: ['RLHF', 'Reward Model', 'PPO'],
      kind: 'alignment',
      position: new THREE.Vector3(8.72, 2.55, 64),
      rotationY: -Math.PI / 2,
      accent: '#14b8a6',
      source: 'InstructGPT / Training language models with human feedback, 2022'
    },
    {
      year: '2024-2025',
      title: 'GRPO：面向推理的组相对优化',
      subtitle: '用一组候选答案的相对表现替代显式价值网络',
      description: 'GRPO 在数学推理和长链思考训练中受到关注。它对同一题采样多条回答，按组内相对奖励估计优势，减少 PPO 中价值模型的负担。',
      bullets: ['同一 prompt 采样 G 个候选 completion', '用组内均值/方差归一化奖励得到相对 advantage', '优化策略时保留 KL 约束，避免模型漂移过快'],
      tags: ['GRPO', 'Reasoning', 'Policy Optimization'],
      kind: 'grpo',
      position: new THREE.Vector3(-8.72, 2.55, 71),
      rotationY: Math.PI / 2,
      accent: '#eab308',
      source: 'DeepSeekMath / DeepSeek-R1 GRPO discussions'
    },
    {
      year: '2024-2026',
      title: '多模态与智能体工作流',
      subtitle: '模型开始看图、听音、写代码、调用工具并完成任务',
      description: '现代 AI 系统不只是一个语言模型，而是模型、检索、工具、记忆、规划和安全策略组成的工作流。',
      bullets: ['视觉、语音、文本统一到跨模态表示', '函数调用和工具使用把模型接到外部系统', 'Agent 工作流强调规划、执行、观察和修正'],
      tags: ['Multimodal', 'Agents', 'Tool Use'],
      kind: 'multimodal',
      position: new THREE.Vector3(8.72, 2.55, 78),
      rotationY: -Math.PI / 2,
      accent: '#60a5fa',
      source: 'Foundation models, multimodal agents, tool-use systems'
    }
  ];

  activeExhibit: MuseumExhibit | null = null;
  isLocked = false;

  private scene?: THREE.Scene;
  private camera?: THREE.PerspectiveCamera;
  private renderer?: THREE.WebGLRenderer;
  private controls?: PointerLockControls;
  private animationId = 0;
  private previousTime = performance.now();
  private readonly velocity = new THREE.Vector3();
  private readonly direction = new THREE.Vector3();
  private readonly moving: MovingState = { forward: false, backward: false, left: false, right: false, sprint: false };
  private readonly exhibitObjects = new Map<THREE.Object3D, MuseumExhibit>();
  private readonly animatedObjects: Array<{ mesh: THREE.Object3D; speed: number; radius?: number }> = [];
  private readonly resizeObserver = new ResizeObserver(() => this.resizeRenderer());

  ngAfterViewInit(): void {
    const host = this.stageRef?.nativeElement;
    if (!host) return;
    this.initScene(host);
    this.animate();
  }

  ngOnDestroy(): void {
    if (this.animationId) cancelAnimationFrame(this.animationId);
    this.resizeObserver.disconnect();
    this.controls?.disconnect();
    this.renderer?.dispose();
    window.removeEventListener('keydown', this.onKeyDown);
    window.removeEventListener('keyup', this.onKeyUp);
  }

  enterMuseum(): void {
    this.controls?.lock();
  }

  private initScene(host: HTMLDivElement): void {
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color('#101827');
    this.scene.fog = new THREE.Fog('#101827', 16, 68);

    this.camera = new THREE.PerspectiveCamera(72, host.clientWidth / Math.max(1, host.clientHeight), 0.1, 120);
    this.camera.position.set(0, CAMERA_HEIGHT, -68);

    this.renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setSize(host.clientWidth, host.clientHeight);
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    host.appendChild(this.renderer.domElement);
    this.resizeObserver.observe(host);

    this.controls = new PointerLockControls(this.camera, this.renderer.domElement);
    this.scene.add(this.controls.object);
    this.controls.addEventListener('lock', () => this.isLocked = true);
    this.controls.addEventListener('unlock', () => this.isLocked = false);

    window.addEventListener('keydown', this.onKeyDown);
    window.addEventListener('keyup', this.onKeyUp);

    this.addLights();
    this.addArchitecture();
    this.addExhibits();
    this.addCentralTimeline();
    this.addWayfinding();
  }

  private addLights(): void {
    if (!this.scene) return;
    this.scene.add(new THREE.HemisphereLight('#dbeafe', '#172033', 1.65));

    const key = new THREE.DirectionalLight('#f8fafc', 1.2);
    key.position.set(-8, 12, -14);
    key.castShadow = true;
    key.shadow.mapSize.set(2048, 2048);
    this.scene.add(key);

    for (let z = -78; z <= 82; z += 12) {
      const light = new THREE.PointLight('#fde68a', 1.6, 18, 1.8);
      light.position.set(0, 5.85, z);
      light.castShadow = true;
      this.scene.add(light);
      this.addCeilingLamp(z);
    }
  }

  private addArchitecture(): void {
    if (!this.scene) return;
    const floor = new THREE.Mesh(
      new THREE.PlaneGeometry(HALL_WIDTH, HALL_LENGTH),
      new THREE.MeshStandardMaterial({
        color: '#d7c7ad',
        roughness: 0.45,
        metalness: 0.04
      })
    );
    floor.rotation.x = -Math.PI / 2;
    floor.receiveShadow = true;
    this.scene.add(floor);

    this.addFloorPattern();
    this.addWall(0, WALL_HEIGHT / 2, -HALL_LENGTH / 2, HALL_WIDTH, WALL_HEIGHT, 0, '#28364a');
    this.addWall(0, WALL_HEIGHT / 2, HALL_LENGTH / 2, HALL_WIDTH, WALL_HEIGHT, Math.PI, '#28364a');
    this.addWall(-HALL_WIDTH / 2, WALL_HEIGHT / 2, 0, HALL_LENGTH, WALL_HEIGHT, Math.PI / 2, '#1f2a3b');
    this.addWall(HALL_WIDTH / 2, WALL_HEIGHT / 2, 0, HALL_LENGTH, WALL_HEIGHT, -Math.PI / 2, '#1f2a3b');

    const ceiling = new THREE.Mesh(
      new THREE.PlaneGeometry(HALL_WIDTH, HALL_LENGTH),
      new THREE.MeshStandardMaterial({ color: '#182233', roughness: 0.7 })
    );
    ceiling.position.y = WALL_HEIGHT;
    ceiling.rotation.x = Math.PI / 2;
    this.scene.add(ceiling);

    for (const z of this.columnPositions()) {
      this.addColumn(-8.42, z);
      this.addColumn(8.42, z);
    }

    for (let z = -66; z <= 74; z += 20) {
      this.addBench(z);
    }
  }

  private addFloorPattern(): void {
    if (!this.scene) return;
    const lineMaterial = new THREE.LineBasicMaterial({ color: '#7f6e55', transparent: true, opacity: 0.18 });
    for (let x = -HALL_WIDTH / 2 + 1; x < HALL_WIDTH / 2; x += 1) {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(x, 0.012, -HALL_LENGTH / 2),
        new THREE.Vector3(x, 0.012, HALL_LENGTH / 2)
      ]);
      this.scene.add(new THREE.Line(geometry, lineMaterial));
    }
    for (let z = -HALL_LENGTH / 2 + 1; z < HALL_LENGTH / 2; z += 1) {
      const geometry = new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-HALL_WIDTH / 2, 0.013, z),
        new THREE.Vector3(HALL_WIDTH / 2, 0.013, z)
      ]);
      this.scene.add(new THREE.Line(geometry, lineMaterial));
    }
  }

  private addWall(x: number, y: number, z: number, width: number, height: number, rotationY: number, color: string): void {
    if (!this.scene) return;
    const wall = new THREE.Mesh(
      new THREE.PlaneGeometry(width, height),
      new THREE.MeshStandardMaterial({ color, roughness: 0.62 })
    );
    wall.position.set(x, y, z);
    wall.rotation.y = rotationY;
    wall.receiveShadow = true;
    this.scene.add(wall);
  }

  private addColumn(x: number, z: number): void {
    if (!this.scene) return;
    const baseMat = new THREE.MeshStandardMaterial({ color: '#c7b99e', roughness: 0.55 });
    const column = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.25, WALL_HEIGHT, 20), baseMat);
    column.position.set(x, WALL_HEIGHT / 2, z);
    column.castShadow = true;
    column.receiveShadow = true;
    this.scene.add(column);

    const top = new THREE.Mesh(new THREE.BoxGeometry(0.68, 0.2, 0.68), baseMat);
    top.position.set(x, WALL_HEIGHT - 0.12, z);
    this.scene.add(top);
  }

  private columnPositions(): number[] {
    const exhibitZ = this.exhibits.map(exhibit => exhibit.position.z);
    const result: number[] = [];
    for (let z = -76; z <= 80; z += 8) {
      const tooClose = exhibitZ.some(exhibit => Math.abs(exhibit - z) < 2.8);
      if (!tooClose) {
        result.push(z);
      }
    }
    return result;
  }

  private addBench(z: number): void {
    if (!this.scene) return;
    const wood = new THREE.MeshStandardMaterial({ color: '#8b5e34', roughness: 0.5 });
    const metal = new THREE.MeshStandardMaterial({ color: '#334155', roughness: 0.4, metalness: 0.25 });
    const seat = new THREE.Mesh(new THREE.BoxGeometry(4.2, 0.24, 0.92), wood);
    seat.position.set(0, 0.62, z);
    seat.castShadow = true;
    this.scene.add(seat);
    [-1.65, 1.65].forEach(x => {
      const leg = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.72, 0.16), metal);
      leg.position.set(x, 0.28, z - 0.24);
      this.scene?.add(leg);
      const leg2 = leg.clone();
      leg2.position.z = z + 0.24;
      this.scene?.add(leg2);
    });
  }

  private addCeilingLamp(z: number): void {
    if (!this.scene) return;
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(1.05, 0.045, 8, 48),
      new THREE.MeshStandardMaterial({ color: '#fde68a', emissive: '#a16207', emissiveIntensity: 0.8 })
    );
    ring.position.set(0, 6.82, z);
    ring.rotation.x = Math.PI / 2;
    this.scene.add(ring);
  }

  private addExhibits(): void {
    for (const exhibit of this.exhibits) {
      const group = new THREE.Group();
      group.position.copy(exhibit.position);
      group.rotation.y = exhibit.rotationY;
      group.userData['exhibitTitle'] = exhibit.title;
      this.exhibitObjects.set(group, exhibit);

      this.addPlaque(group, exhibit);
      this.addArtifact(group, exhibit);
      this.addSpotlight(exhibit);
      this.scene?.add(group);
    }
  }

  private addPlaque(group: THREE.Group, exhibit: MuseumExhibit): void {
    const panel = new THREE.Mesh(
      new THREE.BoxGeometry(4.2, 3.05, 0.14),
      new THREE.MeshStandardMaterial({ color: '#f8fafc', roughness: 0.32, metalness: 0.02 })
    );
    panel.castShadow = true;
    panel.receiveShadow = true;
    group.add(panel);

    const canvas = this.createPlaqueCanvas(exhibit);
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const face = new THREE.Mesh(
      new THREE.PlaneGeometry(3.92, 2.74),
      new THREE.MeshBasicMaterial({ map: texture })
    );
    face.position.z = 0.076;
    group.add(face);

    const frame = new THREE.Mesh(
      new THREE.BoxGeometry(4.46, 3.28, 0.2),
      new THREE.MeshStandardMaterial({ color: exhibit.accent, roughness: 0.28, metalness: 0.35 })
    );
    frame.position.z = -0.04;
    group.add(frame);
    frame.renderOrder = -1;
  }

  private createPlaqueCanvas(exhibit: MuseumExhibit): HTMLCanvasElement {
    const canvas = document.createElement('canvas');
    canvas.width = 1200;
    canvas.height = 840;
    const ctx = canvas.getContext('2d');
    if (!ctx) return canvas;

    ctx.fillStyle = '#f8fafc';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.fillStyle = exhibit.accent;
    ctx.fillRect(0, 0, 28, canvas.height);
    ctx.fillStyle = '#0f172a';
    ctx.font = '900 82px Segoe UI, sans-serif';
    ctx.fillText(exhibit.year, 74, 118);
    ctx.fillStyle = exhibit.accent;
    ctx.font = '800 42px Segoe UI, sans-serif';
    this.wrapText(ctx, exhibit.title, 74, 190, 920, 52, 2);
    ctx.fillStyle = '#475569';
    ctx.font = '600 30px Segoe UI, sans-serif';
    this.wrapText(ctx, exhibit.subtitle, 74, 292, 940, 42, 2);
    ctx.fillStyle = '#1e293b';
    ctx.font = '500 28px Segoe UI, sans-serif';
    this.wrapText(ctx, exhibit.description, 74, 400, 940, 40, 4);

    let bulletY = 610;
    ctx.font = '700 26px Segoe UI, sans-serif';
    for (const bullet of exhibit.bullets) {
      ctx.fillStyle = exhibit.accent;
      ctx.beginPath();
      ctx.arc(90, bulletY - 9, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#0f172a';
      this.wrapText(ctx, bullet, 114, bulletY, 860, 34, 1);
      bulletY += 50;
    }

    ctx.fillStyle = '#64748b';
    ctx.font = '500 22px Segoe UI, sans-serif';
    this.wrapText(ctx, `资料线索：${exhibit.source}`, 74, 790, 1010, 28, 1);
    this.paintFormulaRibbon(ctx, exhibit);
    this.paintMiniIllustration(ctx, exhibit);
    return canvas;
  }

  private paintFormulaRibbon(ctx: CanvasRenderingContext2D, exhibit: MuseumExhibit): void {
    const formulas: Partial<Record<ExhibitKind, string>> = {
      logic: exhibit.year === '1950' ? 'Imitation Game: text-only behavior as evidence' : 'y = 1[Σwᵢxᵢ ≥ θ]',
      perceptron: 'ŷ = sign(w·x + b),  w ← w + η(y - ŷ)x',
      eliza: 'keyword → decomposition rule → reassembly rule',
      expert: 'IF condition₁ ∧ condition₂ THEN conclusion',
      hopfield: 'E = -1/2 Σᵢⱼ wᵢⱼsᵢsⱼ + Σᵢ θᵢsᵢ',
      qlearning: 'Q ← Q + α[r + γ max Q(s′,a′) - Q]',
      svm: 'min 1/2||w||²,  yᵢ(w·xᵢ+b) ≥ 1',
      backprop: '∂L/∂Wˡ = δˡ(aˡ⁻¹)ᵀ',
      cnn: 'feature map = σ(K * X + b)',
      rnn: 'hₜ = f(Wxₜ + Uhₜ₋₁)',
      deep: 'ReLU(x)=max(0,x),  Dropout: h ⊙ mask',
      yolo: 'S×S×(B·5+C): box + objectness + class',
      rl: 'Âₜ = rₜ + γV(sₜ₊₁) - V(sₜ)',
      transformer: 'Attention(Q,K,V)=softmax(QKᵀ/√d)V',
      llm: 'next-token pretraining: maximize Σ log p(xₜ|x<ₜ)',
      alignment: 'RLHF: SFT → Reward Model → PPO',
      grpo: 'Aᵢ = (rᵢ - mean(group)) / std(group)',
      multimodal: 'shared embedding space: image/audio/text/tool tokens'
    };
    const formula = formulas[exhibit.kind];
    if (!formula) return;
    ctx.save();
    ctx.fillStyle = 'rgba(15, 23, 42, .92)';
    ctx.fillRect(74, 726, 1010, 44);
    ctx.fillStyle = '#f8fafc';
    ctx.font = '700 24px Consolas, Menlo, monospace';
    ctx.fillText(formula, 96, 755);
    ctx.restore();
  }

  private paintMiniIllustration(ctx: CanvasRenderingContext2D, exhibit: MuseumExhibit): void {
    const x = 860;
    const y = 74;
    ctx.save();
    ctx.translate(x, y);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(0, 0, 250, 180);
    ctx.strokeStyle = exhibit.accent;
    ctx.lineWidth = 8;
    ctx.strokeRect(0, 0, 250, 180);

    if (exhibit.year === '1950') {
      this.paintTuringPortrait(ctx, exhibit.accent);
    } else if (exhibit.kind === 'perceptron') {
      this.paintPerceptronMachine(ctx, exhibit.accent);
    } else if (exhibit.kind === 'eliza') {
      this.paintElizaTerminal(ctx, exhibit.accent);
    } else if (exhibit.kind === 'expert') {
      this.paintExpertRules(ctx, exhibit.accent);
    } else if (exhibit.kind === 'hopfield') {
      this.paintHopfieldEnergy(ctx, exhibit.accent);
    } else if (exhibit.kind === 'qlearning') {
      this.paintQTable(ctx, exhibit.accent);
    } else if (exhibit.kind === 'svm') {
      this.paintSvmDiagram(ctx, exhibit.accent);
    } else if (exhibit.kind === 'cnn') {
      this.paintCnnDiagram(ctx, exhibit.accent);
    } else if (exhibit.kind === 'rnn') {
      this.paintRnnDiagram(ctx, exhibit.accent);
    } else if (exhibit.kind === 'yolo') {
      this.paintYoloDiagram(ctx, exhibit.accent);
    } else if (exhibit.kind === 'rl') {
      this.paintRlDiagram(ctx, exhibit.accent);
    } else if (exhibit.kind === 'alignment') {
      this.paintRlhfDiagram(ctx, exhibit.accent);
    } else if (exhibit.kind === 'grpo') {
      this.paintGrpoDiagram(ctx, exhibit.accent);
    } else if (exhibit.kind === 'transformer') {
      ctx.strokeStyle = '#f97316';
      ctx.lineWidth = 5;
      for (let i = 0; i < 4; i += 1) {
        ctx.strokeRect(28 + i * 43, 25 + i * 8, 46, 120 - i * 10);
      }
      ctx.fillStyle = '#0f172a';
      ctx.font = '800 24px Segoe UI';
      ctx.fillText('Q K V', 74, 100);
    } else if (exhibit.kind === 'llm') {
      ctx.fillStyle = '#10b981';
      for (let i = 0; i < 7; i += 1) {
        ctx.fillRect(24 + i * 29, 42 + Math.sin(i) * 14, 20, 80);
      }
      ctx.fillStyle = '#0f172a';
      ctx.font = '800 24px Segoe UI';
      ctx.fillText('tokens', 76, 152);
    } else {
      const nodes = [[42, 52], [42, 126], [124, 40], [124, 92], [124, 144], [210, 92]];
      ctx.strokeStyle = '#334155';
      ctx.lineWidth = 3;
      for (const a of nodes.slice(0, 2)) {
        for (const b of nodes.slice(2, 5)) {
          ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.stroke();
        }
      }
      for (const b of nodes.slice(2, 5)) {
        ctx.beginPath(); ctx.moveTo(b[0], b[1]); ctx.lineTo(210, 92); ctx.stroke();
      }
      ctx.fillStyle = exhibit.accent;
      for (const n of nodes) {
        ctx.beginPath(); ctx.arc(n[0], n[1], 14, 0, Math.PI * 2); ctx.fill();
      }
    }
    ctx.restore();
  }

  private paintTuringPortrait(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.fillStyle = '#dbeafe';
    ctx.fillRect(34, 24, 182, 132);
    ctx.fillStyle = '#1e293b';
    ctx.beginPath(); ctx.arc(92, 74, 28, 0, Math.PI * 2); ctx.fill();
    ctx.fillRect(65, 106, 54, 48);
    ctx.fillStyle = accent;
    ctx.fillRect(136, 46, 52, 72);
    ctx.fillStyle = '#fff';
    ctx.font = '900 22px Segoe UI';
    ctx.fillText('Turing', 120, 150);
    ctx.fillStyle = '#0f172a';
    ctx.font = '800 18px Consolas';
    ctx.fillText('Can machines think?', 24, 176);
  }

  private paintPerceptronMachine(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 4;
    ctx.strokeRect(30, 38, 92, 92);
    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(36, 44, 80, 80);
    ctx.fillStyle = accent;
    for (let row = 0; row < 4; row += 1) {
      for (let col = 0; col < 4; col += 1) {
        ctx.fillRect(45 + col * 16, 53 + row * 16, 9, 9);
      }
    }
    ctx.strokeStyle = accent;
    ctx.beginPath(); ctx.moveTo(126, 84); ctx.lineTo(190, 84); ctx.stroke();
    ctx.strokeRect(190, 58, 34, 52);
    ctx.fillStyle = '#0f172a';
    ctx.font = '800 18px Segoe UI';
    ctx.fillText('Mark I style', 55, 166);
  }

  private paintSvmDiagram(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 3;
    ctx.beginPath(); ctx.moveTo(38, 142); ctx.lineTo(205, 26); ctx.stroke();
    ctx.setLineDash([8, 7]);
    ctx.beginPath(); ctx.moveTo(24, 112); ctx.lineTo(182, 18); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(70, 166); ctx.lineTo(230, 58); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = accent;
    [[54, 42], [82, 66], [110, 44], [142, 62]].forEach(([x, y]) => {
      ctx.beginPath(); ctx.arc(x, y, 8, 0, Math.PI * 2); ctx.fill();
    });
    ctx.fillStyle = '#ef4444';
    [[112, 126], [148, 112], [178, 138], [202, 108]].forEach(([x, y]) => {
      ctx.fillRect(x - 7, y - 7, 14, 14);
    });
    ctx.fillStyle = '#0f172a';
    ctx.font = '800 21px Segoe UI';
    ctx.fillText('max margin', 63, 158);
  }

  private paintElizaTerminal(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.fillStyle = '#0f172a';
    ctx.fillRect(26, 32, 198, 118);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 4;
    ctx.strokeRect(26, 32, 198, 118);
    ctx.fillStyle = '#bbf7d0';
    ctx.font = '700 16px Consolas';
    ctx.fillText('USER: I feel sad', 42, 68);
    ctx.fillText('ELIZA: Why sad?', 42, 102);
    ctx.fillText('> keyword: feel', 42, 132);
  }

  private paintExpertRules(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.fillStyle = '#fef3c7';
    ctx.fillRect(28, 30, 190, 126);
    ctx.strokeStyle = accent;
    ctx.lineWidth = 4;
    ctx.strokeRect(28, 30, 190, 126);
    ctx.fillStyle = '#0f172a';
    ctx.font = '800 17px Consolas';
    ['IF fever', 'AND infection', 'THEN diagnosis', 'WITH confidence'].forEach((text, index) => {
      ctx.fillText(text, 48, 62 + index * 27);
    });
  }

  private paintHopfieldEnergy(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.strokeStyle = accent;
    ctx.lineWidth = 4;
    const nodes = [[58, 62], [118, 42], [178, 64], [86, 124], [160, 124]];
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        ctx.beginPath(); ctx.moveTo(nodes[i][0], nodes[i][1]); ctx.lineTo(nodes[j][0], nodes[j][1]); ctx.stroke();
      }
    }
    ctx.fillStyle = '#0f172a';
    nodes.forEach(([x, y]) => { ctx.beginPath(); ctx.arc(x, y, 10, 0, Math.PI * 2); ctx.fill(); });
    ctx.fillStyle = accent;
    ctx.font = '800 20px Consolas';
    ctx.fillText('E ↓', 104, 96);
  }

  private paintQTable(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 3;
    ctx.font = '800 17px Consolas';
    for (let row = 0; row < 4; row += 1) {
      for (let col = 0; col < 4; col += 1) {
        ctx.fillStyle = row === 2 && col === 2 ? accent : '#e2e8f0';
        ctx.fillRect(42 + col * 38, 36 + row * 28, 36, 26);
        ctx.strokeRect(42 + col * 38, 36 + row * 28, 36, 26);
      }
    }
    ctx.fillStyle = '#0f172a';
    ctx.fillText('Q(s,a)', 84, 170);
  }

  private paintCnnDiagram(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.strokeStyle = accent;
    ctx.lineWidth = 4;
    for (let i = 0; i < 4; i += 1) {
      ctx.strokeRect(24 + i * 10, 30 + i * 8, 62, 62);
    }
    for (let i = 0; i < 5; i += 1) {
      ctx.strokeRect(126 + i * 8, 42 + i * 6, 48, 48);
    }
    ctx.fillStyle = '#334155';
    ctx.font = '800 20px Segoe UI';
    ctx.fillText('conv', 46, 128);
    ctx.fillText('maps', 134, 128);
    ctx.beginPath(); ctx.moveTo(95, 72); ctx.lineTo(122, 72); ctx.stroke();
  }

  private paintRnnDiagram(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 4;
    ctx.fillStyle = accent;
    for (let i = 0; i < 4; i += 1) {
      const x = 40 + i * 50;
      ctx.fillRect(x, 70, 32, 32);
      ctx.strokeRect(x, 70, 32, 32);
      if (i < 3) {
        ctx.beginPath(); ctx.moveTo(x + 34, 86); ctx.lineTo(x + 48, 86); ctx.stroke();
      }
    }
    ctx.fillStyle = '#0f172a';
    ctx.font = '800 19px Segoe UI';
    ctx.fillText('hₜ₋₁ → hₜ → hₜ₊₁', 38, 135);
  }

  private paintYoloDiagram(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2;
    for (let i = 0; i <= 5; i += 1) {
      ctx.beginPath(); ctx.moveTo(30 + i * 32, 24); ctx.lineTo(30 + i * 32, 184); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(30, 24 + i * 32); ctx.lineTo(190, 24 + i * 32); ctx.stroke();
    }
    ctx.strokeStyle = accent;
    ctx.lineWidth = 6;
    ctx.strokeRect(72, 64, 78, 58);
    ctx.strokeStyle = '#22c55e';
    ctx.strokeRect(128, 103, 44, 54);
    ctx.fillStyle = '#0f172a';
    ctx.font = '800 20px Segoe UI';
    ctx.fillText('boxes + class', 62, 172);
  }

  private paintRlDiagram(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.strokeStyle = accent;
    ctx.lineWidth = 5;
    ctx.strokeRect(26, 58, 72, 54);
    ctx.strokeRect(154, 58, 72, 54);
    ctx.beginPath(); ctx.moveTo(100, 72); ctx.lineTo(151, 72); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(154, 100); ctx.lineTo(101, 100); ctx.stroke();
    ctx.fillStyle = '#0f172a';
    ctx.font = '800 20px Segoe UI';
    ctx.fillText('agent', 38, 91);
    ctx.fillText('env', 174, 91);
    ctx.fillText('action →', 105, 62);
    ctx.fillText('reward ←', 96, 132);
  }

  private paintRlhfDiagram(ctx: CanvasRenderingContext2D, accent: string): void {
    const boxes = ['SFT', 'RM', 'PPO'];
    ctx.font = '900 24px Segoe UI';
    boxes.forEach((text, index) => {
      const x = 24 + index * 74;
      ctx.fillStyle = index === 1 ? '#f59e0b' : accent;
      ctx.fillRect(x, 66, 54, 48);
      ctx.fillStyle = '#fff';
      ctx.fillText(text, x + 6, 98);
      if (index < 2) {
        ctx.strokeStyle = '#334155';
        ctx.lineWidth = 4;
        ctx.beginPath(); ctx.moveTo(x + 58, 90); ctx.lineTo(x + 70, 90); ctx.stroke();
      }
    });
    ctx.fillStyle = '#0f172a';
    ctx.font = '700 18px Segoe UI';
    ctx.fillText('human preference', 52, 146);
  }

  private paintGrpoDiagram(ctx: CanvasRenderingContext2D, accent: string): void {
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 3;
    ctx.fillStyle = accent;
    for (let i = 0; i < 5; i += 1) {
      ctx.fillRect(36 + i * 34, 42 + (i % 2) * 20, 24, 70);
    }
    ctx.strokeRect(26, 30, 190, 110);
    ctx.fillStyle = '#0f172a';
    ctx.font = '800 20px Segoe UI';
    ctx.fillText('group samples', 54, 166);
    ctx.fillText('Aᵢ = (rᵢ - μ)/σ', 48, 24);
  }

  private addArtifact(group: THREE.Group, exhibit: MuseumExhibit): void {
    const artifact = new THREE.Group();
    artifact.position.set(0, -2.0, 1.1);
    artifact.scale.setScalar(0.86);
    group.add(artifact);

    if (exhibit.kind === 'svm') {
      this.addSvmArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'eliza') {
      this.addElizaArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'expert') {
      this.addExpertArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'hopfield') {
      this.addHopfieldArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'qlearning') {
      this.addQlearningArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'cnn') {
      this.addCnnArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'rnn') {
      this.addRnnArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'yolo') {
      this.addYoloArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'rl') {
      this.addRlArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'alignment') {
      this.addRlhfArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'grpo') {
      this.addGrpoArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'transformer') {
      this.addAttentionArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'llm') {
      this.addTokenArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'multimodal') {
      this.addMultimodalArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'winter') {
      this.addWinterArtifact(artifact);
      return;
    }
    if (exhibit.kind === 'backprop') {
      this.addBackpropArtifact(artifact, exhibit.accent);
      return;
    }
    if (exhibit.kind === 'deep') {
      this.addDeepLearningArtifact(artifact, exhibit.accent);
      return;
    }
    this.addNeuronArtifact(artifact, exhibit.accent);
  }

  private addNeuronArtifact(group: THREE.Group, accent: string): void {
    const material = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.3, metalness: 0.18, emissive: accent, emissiveIntensity: 0.12 });
    const nodePositions = [
      new THREE.Vector3(-1.3, 0.5, 0), new THREE.Vector3(-1.3, -0.5, 0),
      new THREE.Vector3(0, 0.72, 0), new THREE.Vector3(0, 0, 0), new THREE.Vector3(0, -0.72, 0),
      new THREE.Vector3(1.3, 0, 0)
    ];
    const nodes = nodePositions.map(pos => {
      const mesh = new THREE.Mesh(new THREE.SphereGeometry(0.14, 20, 14), material);
      mesh.position.copy(pos);
      mesh.castShadow = true;
      group.add(mesh);
      return mesh;
    });
    const lineMaterial = new THREE.LineBasicMaterial({ color: '#cbd5e1', transparent: true, opacity: 0.68 });
    for (const a of nodes.slice(0, 2)) {
      for (const b of nodes.slice(2, 5)) this.addLine(group, a.position, b.position, lineMaterial);
    }
    for (const b of nodes.slice(2, 5)) this.addLine(group, b.position, nodes[5].position, lineMaterial);
    this.animatedObjects.push({ mesh: group, speed: 0.45 });
  }

  private addDeepLearningArtifact(group: THREE.Group, accent: string): void {
    const mat = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.38, metalness: 0.22 });
    for (let layer = 0; layer < 5; layer += 1) {
      const count = 3 + (layer % 2);
      for (let i = 0; i < count; i += 1) {
        const cube = new THREE.Mesh(new THREE.BoxGeometry(0.28, 0.28, 0.08), mat);
        cube.position.set((layer - 2) * 0.48, (i - (count - 1) / 2) * 0.38, 0);
        cube.castShadow = true;
        group.add(cube);
      }
    }
    this.animatedObjects.push({ mesh: group, speed: -0.28 });
  }

  private addBackpropArtifact(group: THREE.Group, accent: string): void {
    this.addDeepLearningArtifact(group, accent);
    const material = new THREE.LineBasicMaterial({ color: '#fef3c7', transparent: true, opacity: 0.9 });
    for (let i = 0; i < 3; i += 1) {
      this.addLine(group, new THREE.Vector3(1.2 - i * 0.55, -0.95, 0.12), new THREE.Vector3(0.82 - i * 0.55, -0.95, 0.12), material);
    }
    const label = this.createSmallLabel('∂L', accent);
    label.position.set(1.55, -0.95, 0.12);
    label.scale.set(0.36, 0.36, 1);
    group.add(label);
  }

  private addSvmArtifact(group: THREE.Group, accent: string): void {
    const planeMat = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.42, metalness: 0.16, transparent: true, opacity: 0.55 });
    const marginMat = new THREE.MeshBasicMaterial({ color: '#f8fafc', transparent: true, opacity: 0.22, side: THREE.DoubleSide });
    const plane = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 1.25), planeMat);
    plane.rotation.z = -0.58;
    group.add(plane);
    [-0.42, 0.42].forEach(offset => {
      const margin = new THREE.Mesh(new THREE.PlaneGeometry(2.2, 0.04), marginMat);
      margin.rotation.z = -0.58;
      margin.position.y = offset;
      group.add(margin);
    });
    const blue = new THREE.MeshStandardMaterial({ color: '#38bdf8' });
    const red = new THREE.MeshStandardMaterial({ color: '#ef4444' });
    [[-0.9, 0.62], [-0.52, 0.88], [-0.18, 0.54], [0.28, -0.58], [0.68, -0.78], [0.94, -0.44]].forEach(([x, y], index) => {
      const mesh = new THREE.Mesh(index < 3 ? new THREE.SphereGeometry(0.09, 16, 12) : new THREE.BoxGeometry(0.15, 0.15, 0.15), index < 3 ? blue : red);
      mesh.position.set(x, y, 0.12);
      mesh.castShadow = true;
      group.add(mesh);
    });
    this.animatedObjects.push({ mesh: group, speed: 0.2 });
  }

  private addElizaArtifact(group: THREE.Group, accent: string): void {
    const screen = new THREE.Mesh(
      new THREE.BoxGeometry(1.55, 0.9, 0.08),
      new THREE.MeshStandardMaterial({ color: '#0f172a', roughness: 0.28, metalness: 0.18, emissive: '#052e16', emissiveIntensity: 0.25 })
    );
    group.add(screen);
    const lines = ['USER', 'I feel...', 'ELIZA', 'Why do you feel?'];
    lines.forEach((text, index) => {
      const label = this.createTextSprite(text, index % 2 ? '#bbf7d0' : accent, 210, 54, 22);
      label.position.set(0, 0.28 - index * 0.18, 0.08);
      label.scale.set(0.95, 0.24, 1);
      group.add(label);
    });
    this.animatedObjects.push({ mesh: group, speed: 0.12 });
  }

  private addExpertArtifact(group: THREE.Group, accent: string): void {
    ['IF', 'AND', 'THEN'].forEach((text, index) => {
      const card = new THREE.Mesh(
        new THREE.BoxGeometry(0.48, 0.36, 0.08),
        new THREE.MeshStandardMaterial({ color: index === 2 ? accent : '#f8fafc', roughness: 0.34, metalness: 0.08 })
      );
      card.position.set((index - 1) * 0.62, 0, 0);
      group.add(card);
      const label = this.createTextSprite(text, index === 2 ? '#fff' : '#0f172a', 160, 80, 32);
      label.position.copy(card.position).add(new THREE.Vector3(0, 0, 0.065));
      label.scale.set(0.42, 0.2, 1);
      group.add(label);
    });
    const lineMat = new THREE.LineBasicMaterial({ color: '#f8fafc', transparent: true, opacity: 0.72 });
    this.addLine(group, new THREE.Vector3(-0.38, 0, 0.02), new THREE.Vector3(-0.08, 0, 0.02), lineMat);
    this.addLine(group, new THREE.Vector3(0.38, 0, 0.02), new THREE.Vector3(0.68, 0, 0.02), lineMat);
    this.animatedObjects.push({ mesh: group, speed: -0.16 });
  }

  private addHopfieldArtifact(group: THREE.Group, accent: string): void {
    const mat = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.32, metalness: 0.2, emissive: accent, emissiveIntensity: 0.1 });
    const lineMat = new THREE.LineBasicMaterial({ color: '#cbd5e1', transparent: true, opacity: 0.46 });
    const nodes = Array.from({ length: 7 }, (_, index) => {
      const angle = index / 7 * Math.PI * 2;
      const node = new THREE.Mesh(new THREE.SphereGeometry(0.105, 16, 12), mat);
      node.position.set(Math.cos(angle) * 0.78, Math.sin(angle) * 0.52, 0);
      group.add(node);
      return node;
    });
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        this.addLine(group, nodes[i].position, nodes[j].position, lineMat);
      }
    }
    const label = this.createTextSprite('E ↓', '#fef3c7', 128, 80, 34);
    label.position.set(0, -0.78, 0.04);
    label.scale.set(0.42, 0.24, 1);
    group.add(label);
    this.animatedObjects.push({ mesh: group, speed: 0.18 });
  }

  private addQlearningArtifact(group: THREE.Group, accent: string): void {
    const grid = new THREE.Group();
    const mat = new THREE.MeshStandardMaterial({ color: '#e2e8f0', roughness: 0.42 });
    const hot = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.32, emissive: accent, emissiveIntensity: 0.12 });
    for (let row = 0; row < 4; row += 1) {
      for (let col = 0; col < 4; col += 1) {
        const cell = new THREE.Mesh(new THREE.BoxGeometry(0.24, 0.24, 0.04), row === 2 && col === 2 ? hot : mat);
        cell.position.set((col - 1.5) * 0.28, (1.5 - row) * 0.28, 0);
        grid.add(cell);
      }
    }
    group.add(grid);
    const label = this.createTextSprite('Q(s,a)', '#0f172a', 180, 70, 28);
    label.position.set(0, -0.76, 0.06);
    label.scale.set(0.56, 0.22, 1);
    group.add(label);
    this.animatedObjects.push({ mesh: group, speed: 0.2 });
  }

  private addCnnArtifact(group: THREE.Group, accent: string): void {
    const mat = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.38, metalness: 0.16, transparent: true, opacity: 0.72 });
    for (let stack = 0; stack < 4; stack += 1) {
      const size = 1.25 - stack * 0.18;
      const layers = 4 + stack;
      for (let i = 0; i < layers; i += 1) {
        const sheet = new THREE.Mesh(new THREE.BoxGeometry(size, size, 0.035), mat);
        sheet.position.set((stack - 1.5) * 0.62, 0, (i - layers / 2) * 0.075);
        sheet.castShadow = true;
        group.add(sheet);
      }
    }
    const kernel = new THREE.Mesh(new THREE.BoxGeometry(0.38, 0.38, 0.08), new THREE.MeshStandardMaterial({ color: '#fef3c7', emissive: '#92400e', emissiveIntensity: 0.18 }));
    kernel.position.set(-1.35, 0.38, 0.34);
    group.add(kernel);
    this.animatedObjects.push({ mesh: group, speed: -0.22 });
  }

  private addRnnArtifact(group: THREE.Group, accent: string): void {
    const mat = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.34, metalness: 0.18 });
    const line = new THREE.LineBasicMaterial({ color: '#f8fafc', transparent: true, opacity: 0.72 });
    let previous: THREE.Mesh | null = null;
    for (let i = 0; i < 5; i += 1) {
      const cell = new THREE.Mesh(new THREE.BoxGeometry(0.34, 0.34, 0.2), mat);
      cell.position.set((i - 2) * 0.56, Math.sin(i * 0.8) * 0.12, 0);
      group.add(cell);
      if (previous) this.addLine(group, previous.position, cell.position, line);
      previous = cell;
    }
    this.animatedObjects.push({ mesh: group, speed: 0.18 });
  }

  private addYoloArtifact(group: THREE.Group, accent: string): void {
    const gridMat = new THREE.LineBasicMaterial({ color: '#cbd5e1', transparent: true, opacity: 0.62 });
    const boxMat = new THREE.LineBasicMaterial({ color: accent, linewidth: 2 });
    const size = 1.8;
    for (let i = 0; i <= 5; i += 1) {
      const p = -size / 2 + i * size / 5;
      this.addLine(group, new THREE.Vector3(p, -size / 2, 0), new THREE.Vector3(p, size / 2, 0), gridMat);
      this.addLine(group, new THREE.Vector3(-size / 2, p, 0), new THREE.Vector3(size / 2, p, 0), gridMat);
    }
    const geometry = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(-0.48, -0.2, 0.05), new THREE.Vector3(0.42, -0.2, 0.05),
      new THREE.Vector3(0.42, 0.36, 0.05), new THREE.Vector3(-0.48, 0.36, 0.05),
      new THREE.Vector3(-0.48, -0.2, 0.05)
    ]);
    group.add(new THREE.Line(geometry, boxMat));
    this.animatedObjects.push({ mesh: group, speed: 0.12 });
  }

  private addRlArtifact(group: THREE.Group, accent: string): void {
    const mat = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.3, metalness: 0.18 });
    const agent = new THREE.Mesh(new THREE.SphereGeometry(0.22, 24, 16), mat);
    const env = new THREE.Mesh(new THREE.BoxGeometry(0.56, 0.42, 0.42), new THREE.MeshStandardMaterial({ color: '#334155', roughness: 0.42 }));
    agent.position.set(-0.75, 0, 0);
    env.position.set(0.75, 0, 0);
    group.add(agent, env);
    const lineMat = new THREE.LineBasicMaterial({ color: '#fef3c7', transparent: true, opacity: 0.85 });
    this.addLine(group, new THREE.Vector3(-0.45, 0.16, 0), new THREE.Vector3(0.42, 0.16, 0), lineMat);
    this.addLine(group, new THREE.Vector3(0.42, -0.16, 0), new THREE.Vector3(-0.45, -0.16, 0), lineMat);
    this.animatedObjects.push({ mesh: group, speed: 0.36 });
  }

  private addRlhfArtifact(group: THREE.Group, accent: string): void {
    ['SFT', 'RM', 'PPO'].forEach((label, index) => {
      const sprite = this.createSmallLabel(label, index === 1 ? '#f59e0b' : accent);
      sprite.position.set((index - 1) * 0.7, 0, 0);
      sprite.scale.set(0.46, 0.46, 1);
      group.add(sprite);
    });
    const lineMat = new THREE.LineBasicMaterial({ color: '#f8fafc', transparent: true, opacity: 0.68 });
    this.addLine(group, new THREE.Vector3(-0.52, 0, -0.02), new THREE.Vector3(-0.18, 0, -0.02), lineMat);
    this.addLine(group, new THREE.Vector3(0.18, 0, -0.02), new THREE.Vector3(0.52, 0, -0.02), lineMat);
    this.animatedObjects.push({ mesh: group, speed: -0.2 });
  }

  private addGrpoArtifact(group: THREE.Group, accent: string): void {
    const mat = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.32, metalness: 0.2 });
    for (let i = 0; i < 6; i += 1) {
      const bar = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.34 + (i % 3) * 0.18, 0.18), mat);
      bar.position.set((i - 2.5) * 0.32, 0, 0);
      group.add(bar);
    }
    const mean = new THREE.Mesh(new THREE.BoxGeometry(2.1, 0.035, 0.035), new THREE.MeshBasicMaterial({ color: '#f8fafc' }));
    mean.position.y = 0.16;
    group.add(mean);
    const label = this.createSmallLabel('A', accent);
    label.position.set(0, -0.62, 0);
    label.scale.set(0.34, 0.34, 1);
    group.add(label);
    this.animatedObjects.push({ mesh: group, speed: 0.28 });
  }

  private addMultimodalArtifact(group: THREE.Group, accent: string): void {
    const colors = ['#60a5fa', '#34d399', '#f59e0b', '#f472b6'];
    const labels = ['图', '文', '声', '工具'];
    colors.forEach((color, index) => {
      const sprite = this.createSmallLabel(labels[index], color);
      const angle = index / colors.length * Math.PI * 2;
      sprite.position.set(Math.cos(angle) * 0.82, Math.sin(angle) * 0.52, 0);
      sprite.scale.set(0.42, 0.42, 1);
      group.add(sprite);
    });
    const core = new THREE.Mesh(
      new THREE.IcosahedronGeometry(0.26, 1),
      new THREE.MeshStandardMaterial({ color: accent, roughness: 0.28, metalness: 0.28, emissive: accent, emissiveIntensity: 0.18 })
    );
    group.add(core);
    const lineMat = new THREE.LineBasicMaterial({ color: '#f8fafc', transparent: true, opacity: 0.7 });
    for (let i = 0; i < colors.length; i += 1) {
      const angle = i / colors.length * Math.PI * 2;
      this.addLine(group, new THREE.Vector3(0, 0, 0), new THREE.Vector3(Math.cos(angle) * 0.62, Math.sin(angle) * 0.4, 0), lineMat);
    }
    this.animatedObjects.push({ mesh: group, speed: 0.42 });
  }

  private addAttentionArtifact(group: THREE.Group, accent: string): void {
    const material = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.26, metalness: 0.34, emissive: accent, emissiveIntensity: 0.18 });
    for (let i = 0; i < 4; i += 1) {
      const torus = new THREE.Mesh(new THREE.TorusGeometry(0.45 + i * 0.18, 0.018, 8, 54), material);
      torus.rotation.x = Math.PI / 2;
      torus.rotation.z = i * 0.6;
      group.add(torus);
    }
    ['Q', 'K', 'V'].forEach((label, index) => {
      const sprite = this.createSmallLabel(label, accent);
      sprite.position.set((index - 1) * 0.64, 0, 0.03);
      group.add(sprite);
    });
    this.animatedObjects.push({ mesh: group, speed: 0.7 });
  }

  private addTokenArtifact(group: THREE.Group, accent: string): void {
    const material = new THREE.MeshStandardMaterial({ color: accent, roughness: 0.32, metalness: 0.18, emissive: accent, emissiveIntensity: 0.16 });
    for (let i = 0; i < 18; i += 1) {
      const block = new THREE.Mesh(new THREE.BoxGeometry(0.18, 0.18, 0.18), material);
      const angle = i * 0.65;
      block.position.set(Math.cos(angle) * 0.72, (i - 9) * 0.08, Math.sin(angle) * 0.72);
      block.castShadow = true;
      group.add(block);
    }
    this.animatedObjects.push({ mesh: group, speed: 0.55 });
  }

  private addWinterArtifact(group: THREE.Group): void {
    const material = new THREE.MeshStandardMaterial({ color: '#94a3b8', roughness: 0.78, metalness: 0.06 });
    const shard = new THREE.Mesh(new THREE.IcosahedronGeometry(0.8, 0), material);
    shard.scale.set(1.2, 0.7, 0.9);
    shard.castShadow = true;
    group.add(shard);
    this.animatedObjects.push({ mesh: group, speed: 0.18 });
  }

  private addLine(group: THREE.Group, from: THREE.Vector3, to: THREE.Vector3, material: THREE.LineBasicMaterial): void {
    const geometry = new THREE.BufferGeometry().setFromPoints([from, to]);
    group.add(new THREE.Line(geometry, material));
  }

  private createSmallLabel(text: string, color: string): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = 128;
    canvas.height = 128;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(64, 64, 42, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = '#fff';
      ctx.font = '900 48px Segoe UI';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, 64, 66);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
    sprite.scale.set(0.45, 0.45, 1);
    return sprite;
  }

  private createTextSprite(text: string, color: string, width: number, height: number, fontSize: number): THREE.Sprite {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.clearRect(0, 0, width, height);
      ctx.fillStyle = color;
      ctx.font = `900 ${fontSize}px Segoe UI, Consolas, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, width / 2, height / 2);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return new THREE.Sprite(new THREE.SpriteMaterial({ map: texture, transparent: true }));
  }

  private addSpotlight(exhibit: MuseumExhibit): void {
    if (!this.scene) return;
    const light = new THREE.SpotLight(exhibit.accent, 3.2, 9, Math.PI / 5, 0.42, 1.2);
    light.position.set(exhibit.position.x * 0.72, 5.4, exhibit.position.z);
    light.target.position.copy(exhibit.position);
    this.scene.add(light);
    this.scene.add(light.target);
  }

  private addCentralTimeline(): void {
    if (!this.scene) return;
    const material = new THREE.MeshStandardMaterial({ color: '#fbbf24', roughness: 0.28, metalness: 0.24, emissive: '#7c2d12', emissiveIntensity: 0.08 });
    const rail = new THREE.Mesh(new THREE.BoxGeometry(0.16, 0.12, HALL_LENGTH - 8), material);
    rail.position.set(0, 0.08, 2);
    this.scene.add(rail);

    for (const exhibit of this.exhibits) {
      const marker = new THREE.Mesh(new THREE.CylinderGeometry(0.22, 0.22, 0.09, 28), material);
      marker.position.set(0, 0.18, exhibit.position.z);
      marker.rotation.x = Math.PI / 2;
      marker.castShadow = true;
      this.scene.add(marker);
      const label = this.createFloorYearLabel(exhibit.year, exhibit.accent);
      label.position.set(0, 0.24, exhibit.position.z + 0.68);
      label.rotation.x = -Math.PI / 2;
      label.rotation.z = Math.PI;
      this.scene.add(label);
    }
  }

  private createFloorYearLabel(text: string, color: string): THREE.Mesh {
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 160;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      ctx.fillStyle = 'rgba(15, 23, 42, .82)';
      ctx.fillRect(0, 0, 512, 160);
      ctx.strokeStyle = color;
      ctx.lineWidth = 8;
      ctx.strokeRect(8, 8, 496, 144);
      ctx.fillStyle = '#fff';
      ctx.font = '900 48px Segoe UI';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(text, 256, 84);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return new THREE.Mesh(
      new THREE.PlaneGeometry(1.7, 0.52),
      new THREE.MeshBasicMaterial({ map: texture, transparent: true })
    );
  }

  private addWayfinding(): void {
    if (!this.scene) return;
    const title = this.createBannerTexture('AI HISTORY MUSEUM', '从早期机器学习到 Transformer 与大模型');
    const banner = new THREE.Mesh(
      new THREE.PlaneGeometry(9, 2.2),
      new THREE.MeshBasicMaterial({ map: title, transparent: true })
    );
    banner.position.set(0, 4.05, -HALL_LENGTH / 2 + 0.14);
    this.scene.add(banner);
  }

  private createBannerTexture(title: string, subtitle: string): THREE.CanvasTexture {
    const canvas = document.createElement('canvas');
    canvas.width = 1400;
    canvas.height = 360;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      const gradient = ctx.createLinearGradient(0, 0, 1400, 0);
      gradient.addColorStop(0, 'rgba(14, 165, 233, .92)');
      gradient.addColorStop(1, 'rgba(245, 158, 11, .9)');
      ctx.fillStyle = gradient;
      ctx.fillRect(0, 0, 1400, 360);
      ctx.fillStyle = '#fff';
      ctx.font = '900 96px Segoe UI';
      ctx.textAlign = 'center';
      ctx.fillText(title, 700, 150);
      ctx.font = '700 42px Segoe UI';
      ctx.fillText(subtitle, 700, 235);
    }
    const texture = new THREE.CanvasTexture(canvas);
    texture.colorSpace = THREE.SRGBColorSpace;
    return texture;
  }

  private wrapText(
    ctx: CanvasRenderingContext2D,
    text: string,
    x: number,
    y: number,
    maxWidth: number,
    lineHeight: number,
    maxLines: number
  ): void {
    const chars = Array.from(text);
    let line = '';
    let lines = 0;
    for (const char of chars) {
      const test = line + char;
      if (ctx.measureText(test).width > maxWidth && line) {
        ctx.fillText(line, x, y + lines * lineHeight);
        line = char;
        lines += 1;
        if (lines >= maxLines) return;
      } else {
        line = test;
      }
    }
    if (line && lines < maxLines) ctx.fillText(line, x, y + lines * lineHeight);
  }

  private onKeyDown = (event: KeyboardEvent): void => {
    switch (event.code) {
      case 'ArrowUp':
      case 'KeyW':
        this.moving.forward = true;
        break;
      case 'ArrowLeft':
      case 'KeyA':
        this.moving.left = true;
        break;
      case 'ArrowDown':
      case 'KeyS':
        this.moving.backward = true;
        break;
      case 'ArrowRight':
      case 'KeyD':
        this.moving.right = true;
        break;
      case 'ShiftLeft':
      case 'ShiftRight':
        this.moving.sprint = true;
        break;
    }
  };

  private onKeyUp = (event: KeyboardEvent): void => {
    switch (event.code) {
      case 'ArrowUp':
      case 'KeyW':
        this.moving.forward = false;
        break;
      case 'ArrowLeft':
      case 'KeyA':
        this.moving.left = false;
        break;
      case 'ArrowDown':
      case 'KeyS':
        this.moving.backward = false;
        break;
      case 'ArrowRight':
      case 'KeyD':
        this.moving.right = false;
        break;
      case 'ShiftLeft':
      case 'ShiftRight':
        this.moving.sprint = false;
        break;
    }
  };

  private animate(): void {
    this.animationId = requestAnimationFrame(() => this.animate());
    const now = performance.now();
    const delta = Math.min(0.05, (now - this.previousTime) / 1000);
    this.previousTime = now;
    this.updateMovement(delta);
    this.updateActiveExhibit();

    const elapsed = now / 1000;
    for (const item of this.animatedObjects) {
      item.mesh.rotation.y += delta * item.speed;
      item.mesh.position.y += Math.sin(elapsed * 1.3 + item.speed) * 0.0008;
    }

    if (this.scene && this.camera) {
      this.renderer?.render(this.scene, this.camera);
    }
  }

  private updateMovement(delta: number): void {
    if (!this.controls?.isLocked || !this.camera) return;
    this.velocity.x -= this.velocity.x * 9.5 * delta;
    this.velocity.z -= this.velocity.z * 9.5 * delta;

    this.direction.z = Number(this.moving.forward) - Number(this.moving.backward);
    this.direction.x = Number(this.moving.right) - Number(this.moving.left);
    this.direction.normalize();

    const speed = this.moving.sprint ? 38 : 24;
    if (this.moving.forward || this.moving.backward) this.velocity.z -= this.direction.z * speed * delta;
    if (this.moving.left || this.moving.right) this.velocity.x -= this.direction.x * speed * delta;

    this.controls.moveRight(-this.velocity.x * delta);
    this.controls.moveForward(-this.velocity.z * delta);

    const object = this.controls.object;
    object.position.x = THREE.MathUtils.clamp(object.position.x, -HALL_WIDTH / 2 + 1.15, HALL_WIDTH / 2 - 1.15);
    object.position.z = THREE.MathUtils.clamp(object.position.z, -HALL_LENGTH / 2 + 2.1, HALL_LENGTH / 2 - 2.1);
    object.position.y = CAMERA_HEIGHT;
  }

  private updateActiveExhibit(): void {
    if (!this.controls) return;
    const position = this.controls.object.position;
    let best: MuseumExhibit | null = null;
    let bestDistance = Number.POSITIVE_INFINITY;
    for (const exhibit of this.exhibits) {
      const distance = position.distanceTo(exhibit.position);
      if (distance < bestDistance) {
        best = exhibit;
        bestDistance = distance;
      }
    }
    this.activeExhibit = bestDistance < 8 ? best : null;
  }

  private resizeRenderer(): void {
    if (!this.stageRef || !this.renderer || !this.camera) return;
    const host = this.stageRef.nativeElement;
    const width = Math.max(1, host.clientWidth);
    const height = Math.max(1, host.clientHeight);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
  }
}
