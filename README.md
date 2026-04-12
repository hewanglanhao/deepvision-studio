# DeepVision Studio (Angular Frontend Prototype)

本目录是课程 PJ「深度学习算法可视化仿真平台」的前端原型实现。

## 技术栈

- Angular 20（Standalone Component）
- HTML5 + CSS + TypeScript
- 纯前端模拟训练流程（当前不依赖后端）

## 已实现功能（首版）

- 网络结构编辑器：拖拽添加层、删除层、编辑层参数、手动连接管理。
- 数据集管理：内置 MNIST/CIFAR-10 子集的可视化预览。
- 训练过程可视化：Loss/Accuracy 曲线实时更新。
- 交互训练控制：开始、暂停、停止，支持 batch/epoch/optimizer/lr 设置。
- 单样本推理演示：展示各层激活条形图。
- 结构影响实验：深度、激活函数、优化器三类对比实验。
- 预设任务与评估：可运行预设任务并生成评估日志。
- 可解释性演示：特征图和 Grad-CAM 热力图模拟。
- 大模型辅助分析占位：前端策略建议面板（后续可接真实 API）。

## 新增增强（第二轮）

- 代码结构拆分：训练模拟与数据结构从页面组件中抽离。
- 算法覆盖增强：优化器扩展到 Adam/AdamW/RMSProp/SGD/Momentum/Nesterov/Adagrad/Adadelta。
- 学习率调度：支持 `none` / `step` / `cosine`。
- 模型模板：内置 MLP、CNN、ResNet Mini、LSTM Lite（教学占位）快速切换。
- 验证指标可视化：新增 Validation Accuracy 曲线。
- 评估信息增强：新增 10x10 混淆矩阵可视化。

## 本地运行

```bash
npm install
npm start
```

访问 `http://localhost:4200/`。

## 构建

```bash
npm run build
```

构建输出位于 `dist/mypj-frontend`。

## 目录说明

- `src/app/app.ts`: 页面状态与交互逻辑。
- `src/app/app.html`: 主界面结构。
- `src/app/app.css`: 页面样式与响应式布局。
- `src/app/sim-models.ts`: 统一类型定义（层、连接、任务、指标、模板）。
- `src/app/sim-engine.ts`: 训练模拟、曲线生成、视觉数据生成、模板与实验逻辑。

## 后续建议

- 接入 Spring Boot + MyBatis 后端，替换当前训练模拟为真实训练结果流。
- 引入 WebSocket 支持多人实时观察训练过程。
- 将 AI 建议面板替换为真实 LLM API，并记录请求日志以满足课程规范。
