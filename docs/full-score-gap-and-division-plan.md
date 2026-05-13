# DeepVision Studio 满分冲刺缺口与分工方案

> 生成时间：2026-05-06  
> 目标：基于当前代码实际情况，识别项目距离课程评分点的缺口，并把后续工作分配到 A/B/C/D 四个模式及全局工程任务中。  
> 原则：不推翻现有 ABCD 分工；四个模式之间不做业务数据联动，避免开发冲突；允许复用组件、工具函数、样式、接口封装和 AI 助手组件；每个模式在自己的数据和运行链路内补齐评分点；最后再做部署与答辩材料整合。

## 0.0 关键协作约束

本项目四个模式由四位成员独立开发。为了避免冲突，后续安排遵循以下边界：

- A/B/C/D **不互传业务数据**：不要求 B 训练出的 checkpoint 给 A/C 使用，也不要求 C 的特征图回传给 A/B。
- 可以共享基础代码：如网络结构类型、UI 组件、AI 助手组件、API client、颜色变量、文档模板。
- 每个模式都要能单独讲清楚自己的学习目标和可得分功能。
- 评分表中看似需要“训练后再推理”的项，可以由 B 模式在自己内部完成；A 模式则负责“单样本前向传播教学演示”，不承担 B 训练模型的推理。
- 最终答辩讲成“四个独立教学实验室组成的平台”，而不是一条跨模式流水线。

## 0. 当前项目真实状态总览

### 0.1 架构现状

当前项目是：

- 前端：Angular 20，路由入口包括 `/mode-a`、`/mode-b`、`/mode-c`、`/mode-d`、`/network-3d`。
- Spring 后端：负责登录注册、JWT、A 模式记录保存、A 模式前向传播代理、B 模式训练任务、WebSocket 指标推送、Checkpoint、LLM 代理接口。
- Python forward 服务：Flask，负责 A/B 页面中的前向传播计算。
- Python training worker：PyTorch，负责 B 模式真实训练。
- Mode C：前端 TensorFlow.js 加载静态 CNN Explainer 模型资源，在浏览器中做样本推理和中间激活解释。
- Mode D：纯前端 TypeScript MLP 前向传播、反向传播、优化器可视化，没有后端。

### 0.2 已验证过的构建情况

已执行：

```powershell
cd frontend
npm run build
```

结果：构建成功，仅有 Angular 模板 optional chaining warning。

已执行：

```powershell
cd backend/spring
mvn test
```

结果：构建成功，但没有测试用例。

已执行：

```powershell
python -m py_compile backend\python-forward\app.py backend\python-forward\forward_engine.py backend\python-training\training_worker.py
```

结果：语法检查通过。

### 0.3 重要风险

这些不是小问题，直接影响满分：

1. **大量中文乱码**
   - `README.md`、`AI_USAGE.md`、`home-page.component.*`、`mode-b` 部分文件、`mode-c/d` 资源文案中有乱码。
   - 这会严重影响“文档全面、详实准确”和“UI 清晰美观”的评分。

2. **B 模式真实训练依赖本地数据集目录**
   - Python worker 读取路径为 `backend/spring/datasets/builtin/...`。
   - 当前仓库没有看到 `backend/spring/datasets` 下的真实数据。
   - 已有脚本 `backend/spring/scripts/download_builtin_datasets.py` 可下载 MNIST/CIFAR/Iris/points，但必须在部署和答辩前执行并验证。

3. **A 模式训练 Mock 残留已清理**
   - A 模式旧版曾保留训练面板和 `trainingSvc.start()` 相关演示逻辑，容易被误判为“假训练”。
   - 目前 A 页面已删除训练区域和相关 TS 状态，主线回到真实前向传播、卷积核影响和单样本推理教学。
   - A 仍不作为真实训练得分来源；真实训练、训练控制和训练后评估应由 B 模式承担。

4. **B 模式训练是真实后端，但实验对比仍是 Mock**
   - B 的 `startTraining()` 调用 `trainingSvc.startBackend()`，是真实 PyTorch 训练。
   - 但 `runExperiments()` 使用 `SimEngine.evaluateTask()` 和 `SimEngine.runExperiment()`，仍是模拟对比。

5. **Mode C 是前端 TF.js 静态模型解释，不是 B 训练模型的解释**
   - C 加载 `/mode-c/cnn-explainer/assets/data/model.json` 和 TF.js 静态资源。
- 它能做真实浏览器推理和中间激活解释，但不使用 B 模式训练出的 checkpoint；这是刻意保持模式独立。
   - 没有真实 Grad-CAM。

6. **Mode D 是本地教学引擎**
   - D 有完整 MLP 前向、loss、反向传播、SGD/Momentum/Adam 更新。
- 但没有后端、没有持久化，也不和 B 模式训练系统联动；它定位为独立的 MLP/反向传播教学模式。

7. **LLM 只接入 A 模式 UI**
   - 后端 `/api/llm/chat` 已有，走火山 Ark 配置。
   - 但必须配置 `ARK_API_KEY`。
   - 当前只有 A 页面挂了 `app-llm-floating-assistant`，B/C/D 未接入。

8. **没有 Docker、docker-compose、公有云部署、Nginx/HTTPS/HTTP2 配置**
   - 这是工程能力 5 分和通用进阶技术分的明显缺口。

9. **没有自动化测试**
   - Spring `mvn test` 没有测试源。
   - 前端也只有默认 spec，不足以证明关键功能稳定。

## 1. 评分项逐条缺口判断

### 1.1 网络结构编辑器

当前完成：

- A/B 都有网络结构编辑器。
- 支持拖拽添加、删除、重排层。
- 支持 Conv2D、Pool2D、Flatten、Dense、Activation、Dropout、Output。
- 支持配置卷积核大小、输出通道、stride、padding、dilation、激活函数、Dense 单元数、Dropout 比率等。
- `network-overview.component.ts` 有 drag/drop 逻辑。

主要缺口：

- 目前更像线性网络编辑器，连接主要由 `SimEngine.rebuildLinearConnections()` 自动重建。
- 没有真正的“任意节点连线/删除边/分支网络”编辑体验；这是进阶结构能力，不等同于基础的添加/删除网络层。
- 不支持 ResNet 残差连接这类分支结构。

建议：

- 基础分层面，A/B 已经可以解释为支持添加、删除、重排并自动连接线性网络层。
- 不建议再让 A 做任意连边或 DAG 编辑器；A 的重点应保持在真实前向传播和卷积核影响。
- 若要体现 ResNet/残差连接，优先由 B 做训练模板或高级结构预设，不要求 A 继续扩展。

负责人建议：A 保持现有线性网络编辑与前向传播演示；B 若承担 ResNet，再在 B 内部实现高级结构模板。

### 1.2 数据集管理

当前完成：

- B 后端注册了 `mnist-1000`、`cifar10-500`、`cifar10-5000`、`iris`、`points-2d`。
- B 前端有数据集列表、样本数、类别数、shape、标签分布、训练/验证/测试划分、预览。
- Spring 提供 `/api/training/datasets/builtin`、`/api/training/datasets/{datasetId}`、`/api/training/datasets/imports`。
- 自定义 CSV/图片导入有前后端逻辑。

主要缺口：

- 当前仓库未见真实内置数据目录，B 训练会因为找不到图像/CSV 而失败。
- 内置数据集注册时有“MNIST 全量/CIFAR 全量”文案，但 ID 仍是 `mnist-1000`、`cifar10-500`，命名不一致，容易被老师追问。
- 图片预览在没有真实数据时会退回 SVG 占位。
- 自定义上传图片只保存预览信息，Python worker 的 `load_dataset()` 只支持内置 datasetId，不支持 upload datasetId，所以“上传图片后真实训练”目前不可用。

建议：

- B 必须先跑通 `python backend/spring/scripts/download_builtin_datasets.py`。
- 修改数据集文案，让 ID、样本数、实际数据一致。
- 如果数据太大，准备 `mnist-small`、`cifar10-small` 两个小子集，保证答辩机器可快速训练。
- 若要保上传数据集得分，需要扩展 Python worker 支持 `upload-*` 数据集路径；时间不够则界面明确写“上传用于预览/结构检查，真实训练使用内置数据集”。

负责人建议：B。

### 1.3 训练过程可视化

当前完成：

- B 有真实 PyTorch 训练。
- Spring 启动 Python worker，WebSocket 推送 metric。
- 前端显示训练损失、验证损失、训练准确率、验证准确率、学习率、梯度范数、权重均值/方差近似直方图、训练日志、测试集结果。

主要缺口：

- 权重直方图后端当前是根据 `weightMean/weightStd` 生成的近似分布，不是真实每层权重分布。
- 逐层反向传播计算图、梯度热力图、真实权重更新在 B 页面中明确标注“教学占位，尚未实现”。
- 对图像任务的“训练过程中特征图实时可视化”没有从 B 的训练模型中取出。

建议：

- B：补一个 `/api/training/{jobId}/activations/sample` 或训练完成后单样本激活接口，返回 Conv 层 feature maps。
- 如果时间不够，C 用 TF.js 静态模型承担“卷积层特征图可视化”，B 专注真实训练曲线。
- 删除或改写“尚未实现”的暴露文案，避免答辩界面自曝短板。

负责人建议：B 负责真实训练曲线；C 负责 CNN feature maps；D 负责反向传播教学可视化。

### 1.4 交互式训练控制

当前完成：

- B 有开始、暂停、恢复、停止、重置。
- 后端通过 control file 控制 Python worker。
- 支持 batch size、epoch、learning rate、optimizer、scheduler、lrDecay。

主要缺口：

- 如果 Python worker 启动路径或 Conda 环境不正确，会无法训练。`application.yml` 默认写了本机路径 `C:/Users/lizihan/miniconda3/envs/dl-platform/python.exe`，换机器会失败。
- 训练状态异常时的用户提示还不够友好。

建议：

- 全局工程：把 Python 路径改为环境变量优先，并在部署文档写清楚。
- B：启动训练前增加“后端健康检查/数据集存在检查/Python worker 检查”。

负责人建议：B + 全局部署负责人。

### 1.5 单样本推理演示

当前完成：

- A 有后端前向传播计算，可输入图片样本或上传图片，展示每层输出 shape、特征图/向量、最终 tensor。
- Python forward 是真实数学计算，但权重来自用户设置/默认卷积核；它是 A 模式自己的前向传播教学，不使用 B 的训练结果。
- B 在训练完成后会在测试集上推理并展示样本预测结果，这是 B 模式内部的“训练后推理/评估”。

主要缺口：

- 评分项要求“训练后，可输入单个样本，可视化其在前向传播过程中各层激活值”。在不跨模式联动的约束下，这一项应由 B 模式在自己内部补齐。
- A 已能做“单样本前向传播教学”，但不是训练后模型。
- B 的 checkpoint 当前只能重新跑测试集，不能在 B 内部选择单张样本并返回逐层激活。

建议：

- A：继续承担“输入单张图片，展示当前结构的前向传播过程”，作为前向传播教学得分点。
- B：在 B 页面内部新增“训练后单样本推理”面板。选择 checkpoint 和样本，后端返回预测结果、各层 shape、Conv feature maps 或 Dense activation。
- 轻量方案：如果逐层激活来不及，B 至少在测试样本卡片中展示单样本预测概率、真实标签、置信度和模型结构摘要；但满分更建议返回逐层激活。

负责人建议：A 负责 A 内部前向传播教学；B 负责 B 内部训练后单样本推理，不跨模式传数据。

### 1.6 结构影响实验

当前完成：

- A/B 都有 `runExperiments()`，但本质是 `SimEngine` 模拟。
- D 可以通过学习率、优化器、网络结构变化观察 MLP loss 变化，但也是本地教学引擎。

主要缺口：

- 评分点要求引导用户做对比实验，观察训练速度和精度影响。若只用模拟，很难拿满。

建议：

- B 新增“真实实验对比”模块：
  - 选择基线配置。
  - 自动派生 2-3 个实验：加深网络、换激活函数、换优化器。
  - 串行启动 PyTorch 训练，限制小数据集和少 epoch。
  - 输出对比表：最终准确率、最终 loss、训练耗时、收敛 epoch。
- D 保留“原理实验对比”，用于解释为什么 Adam/Momentum/学习率会影响收敛。

负责人建议：B 主做真实对比；D 做教学解释。

### 1.7 预设任务与评估

当前完成：

- B 有 MNIST、CIFAR-10、Iris、2D points。
- 训练完成后 Python worker 会在测试集评估准确率、loss，返回样本预测。
- Checkpoint 可保存和重新跑测试集。

主要缺口：

- 回归任务是占位。
- 自定义上传数据集不能被 Python worker 真实训练。
- 数据文件缺失会导致内置任务跑不通。

建议：

- 若时间紧，删除或隐藏“回归任务（占位）”，避免扣“完成度”。
- 确保至少 3 个分类任务真实可跑：MNIST small、Iris、points-2d。CIFAR 可作为可选高耗时任务。
- 准备演示用默认配置，避免现场调参失败。

负责人建议：B。

### 1.8 大模型辅助分析

当前完成：

- Spring 有 `/api/llm/chat`。
- A 页面挂了 AI 浮动助手，并可带 A 模式上下文。
- 后端使用 Ark 兼容 chat completions，必须配置 `ARK_API_KEY`。

主要缺口：

- B/C/D 页面未接入 AI 助手。
- LLM 错误处理/未配置 key 时的提示不足。
- 没有专门的“调参建议/训练现象解释/卷积解释/反向传播解释”模板。

建议：

- 全局复用 `LlmFloatingAssistantComponent`，分别给 B/C/D 提供 contextProvider。
- B：提供训练曲线、超参数、数据集、模型结构上下文，让 AI 给调参建议。
- C：提供当前样本、预测类别、选中卷积层、通道统计，让 AI 解释 CNN 关注什么。
- D：提供当前 loss、梯度、优化器、选中神经元，让 AI 解释反向传播和优化器。
- 后端补一个 `/api/llm/health` 或前端捕获 `ARK_API_KEY is not configured` 并显示配置说明。

负责人建议：全局 AI 负责人，可由 A 或较轻松成员做；各模式提供自己的 contextProvider。

### 1.9 进阶功能

当前完成：

- Web3D：有 Three.js 3D 网络查看器 `/network-3d`，A 有入口。
- WebSocket：B 训练指标可多客户端订阅同一个 job。
- 高级训练技术：B 支持 Adam、AdamW、SGD、Momentum、Nesterov、RMSProp、Adagrad、Adadelta；支持 StepLR、CosineAnnealingLR。
- D 支持 SGD、Momentum、Adam 的教学级优化器可视化。
- A 已补强 forward-only 教学能力：层公式解释、Dense/Output 权重来源声明、真实后端卷积核对比面板。

主要缺口：

- 无真实 Grad-CAM。
- 无 ResNet/LSTM 等复杂结构的完整教学或训练展示。
- WebSocket 没有“在线讨论/协作房间”，只是训练指标广播。
- 无 HTTP/2/HTTP/3、WASM、公有云性能配置。

建议：

- C 优先做真实或准真实 Grad-CAM，这是最贴近课程题目的进阶功能。
- ResNet 优先交给 B：做“ResNet Block 模板 + 真实训练/评估”最能支撑“高级网络与训练技术”得分。若时间不足，B 可以先做小型 residual CNN 预设，而不是完整通用 DAG 编辑器。
- A 不再承担 Residual Add/Skip Connection 教学演示，避免和 B 的 ResNet 进阶功能重复。A 保持 forward-only 的 CNN/MLP 单样本解释定位。
- LSTM 不建议放在 A。A 当前围绕图像、卷积核和 CNN 前向传播；LSTM 需要序列输入、时间步、hidden state/cell state/gate UI，放进 A 会割裂主题。
- LSTM 更适合 D：做一个“LSTM Cell 可视化”子页或面板，展示 input/forget/output gate、candidate、cell state、hidden state 随时间步更新。若要真实训练版序列任务，则放 B，但成本高于 D 的教学可视化。
- WebSocket 聊天/讨论房间更适合放在 B：围绕同一个训练 job 观察指标并讨论训练过程。它名义上属于“实时协作”进阶项，但不建议做成全局强依赖功能。
- AI 当前先不做全局统一报告生成器；在 B/C/D 接口未稳定前，A 只完善自己的 AI 预设问题和上下文解释。
- 部署阶段加 Nginx HTTPS/HTTP2，写入部署文档。

负责人建议：A 做好前向传播、公式解释、卷积核对比和 A 内 AI；B 做真实训练、结构影响实验、训练后推理和 ResNet；C 做 Grad-CAM；D 做反向传播、优化器和可选 LSTM Cell；协作聊天室若做则放 B，部署最后统一做。各模式只共享组件和类型，不共享训练结果。

## 2. A 模式：前向传播与卷积核影响

### 2.1 当前 A 模式真实功能

A 模式当前适合定位为“模型如何做一次前向传播”：

- 有图片样本选择、图片上传、预处理显示。
- 有网络模板、拖拽添加/删除/重排层。
- 有 Conv/Pool/Dense/Activation/Dropout 参数编辑。
- 有卷积核矩阵和经典 kernel preset。
- 有 Spring -> Python forward 后端真实计算。
- 有每层输出 shape、特征图/向量可视化。
- 有层公式解释卡片，覆盖 Conv/Pool/Flatten/Dense/Activation/Dropout 等前向计算逻辑。
- Dense/Output 已声明权重来源：使用按层 ID 生成的确定性演示权重，当前只开放偏置编辑，避免庞大矩阵干扰教学。
- 有卷积核对比弹窗：对当前 Conv 层和当前输出通道应用多种经典卷积核，调用真实 forward 后端生成对比特征图和统计值。
- 有 A 模式历史记录保存/恢复。
- 有 3D 网络查看器入口。
- 有 AI 浮动助手。
- 旧版训练 Mock 区域已删除，A 页面不再暴露训练控制残留。

### 2.2 A 模式存在的问题

1. A 不负责真实训练，这一点需要在答辩表达中讲清楚。
   - 这是四模式独立分工的结果，不是缺陷。
   - A 的价值是“让用户看懂一次前向传播和卷积核如何改变特征图”，训练、暂停、评估、checkpoint 由 B 承担。

2. A 的 forward 权重不是训练后模型权重。
   - 这是符合分工边界的：A 只解释“前向传播怎么算”，不负责展示 B 训练出的权重。
   - Dense/Output 已经在界面中说明权重来源；卷积核和 bias 仍由用户手动控制，便于观察直观影响。

3. 网络连接能力偏线性。
   - A 已支持添加、删除、重排层，并自动形成线性连接，足以支撑基础网络编辑器演示。
   - 任意连边、分支、残差连接属于进阶结构能力；若 B 做 ResNet，这部分不再要求 A 承担。

4. 卷积核对比依赖 forward 后端。
   - Python forward 服务未启动时，对比弹窗会失败；答辩前需要确认错误提示清楚。

5. 中文乱码风险。
   - A 页面相对较好，但仍要检查所有弹窗、帮助文档、记录区文案。

### 2.3 A 负责人后续任务

#### A-P0：稳定前向传播演示

必须完成：

- 固化一条 A 模式答辩流程：
  1. 选择图片。
  2. 选择 CNN Classic。
  3. 修改卷积核为 Edge/Blur/Sharpen。
  4. 点击“开始计算”。
  5. 展示每层 shape、特征图和最终向量。
  6. 打开卷积核对比面板，对比 Identity/Edge/Sharpen/Blur/Sobel 等真实 forward 结果。
  7. 选中层检查器，展示公式解释。
  8. 保存记录，再从历史记录恢复。
  9. 打开 3D 网络查看器。
- 保证 Python forward 服务未启动时有清晰错误提示。
- 已完成：删除 A 的训练 Mock 区域和相关 TS 残留，避免影响答辩判断。

#### A-P1：强化“单样本前向传播教学”

在不接收 B 数据的前提下，A 要把自己的单样本前向传播做完整：

- 输入一张内置图片或上传图片。
- 使用当前 A 模式网络结构和参数计算。
- 对每一层展示：
  - 输入/输出 shape。
  - Conv/Pool 的二维或多通道特征图。
  - Dense/Output 的激活向量和 top-k。
  - 当前层公式解释。
  - 当前层参数和权重来源说明。
- 已完成：卷积核对比面板，用弹窗方式避免挤压原本单样本传播面板。
- 暂不建议继续做“逐层播放”：图片小、计算快，答辩观感不如“层检查器 + 公式解释 + 卷积核对比”直观。
- AI 助手可解释当前选中层，但只解释 A 当前前向结果。

答辩表述：

- A 模式满足“单样本推理演示”和“前向传播过程可视化”的教学目标。
- “训练后模型的单样本推理”由 B 模式在 B 内部完成。

#### A-P1：网络编辑器答辩口径整理

A 当前已经支持基础网络编辑器能力：添加、删除、重排层，并由系统自动形成线性连接。后续不建议继续在 A 做任意连边、DAG 或 Residual Add：

- 答辩时表述为：“A 模式聚焦线性 CNN/MLP 的单样本前向传播，支持拖拽调整层顺序并自动重建连接。”
- 如果老师追问 ResNet/分支结构，说明完整 ResNet 由 B 模式作为高级训练结构体现。
- 如果老师追问 LSTM，说明 LSTM 更适合 D 模式做时间步和门控状态可视化。

#### A-P2：A 模式 AI 助手增强

当前已经接入 AI，但建议增强预设问题：

卷积核与特征图：

- “解释当前选中卷积核为什么会产生这个特征图。”
- “对比这些卷积核结果，说明 Edge/Blur/Sharpen/Sobel 的差异。”
- “为什么边缘检测卷积核会让轮廓更明显？”
- “为什么模糊卷积核会让图像细节变少？”
- “当前输出通道和其他通道可能关注了什么不同特征？”

shape 与公式：

- “解释当前层输出 shape 如何计算。”
- “把当前层公式用初学者语言解释一遍。”
- “Conv 的 stride、padding、kernel size 分别如何影响输出大小？”
- “Flatten 为什么会把特征图变成一维向量？”
- “Dense/Output 的确定性演示权重是什么意思，为什么这里不手动编辑完整矩阵？”

网络配置诊断：

- “根据当前网络结构指出可能的配置错误。”
- “为什么这个网络最后输出维度是 10？”
- “Dropout 在 A 模式里为什么只是前向传播演示，不代表训练过程？”
- “如果我想让特征图更平滑/更锐利，应该调整哪个卷积核？”

答辩讲解：

- “用 1 分钟讲清楚 A 模式展示了什么。”
- “用初学者语言解释这次前向传播。”
- “帮我总结当前样本从输入到输出经历了哪些层。”
- “帮我准备老师追问：A 为什么不展示训练后的权重？”

## 3. B 模式：真实训练、数据集、评估

### 3.1 当前 B 模式真实功能

B 是最接近评分表核心要求的模式：

- 有 Spring 后端训练任务。
- 有 Python PyTorch worker。
- 有 WebSocket 指标流。
- 有数据集接口和数据集导入接口。
- 有训练控制：开始、暂停、恢复、停止、重置。
- 有 batch size、epoch、learning rate、optimizer、scheduler。
- 有测试集评估。
- 有 checkpoint 保存与重新测试。
- 有训练曲线、日志、测试样本预测结果。

### 3.2 B 模式存在的问题

1. 真实数据集可能缺失。
   - 这是 B 模式最大风险。没有 `datasets/builtin`，worker 会报错。

2. 默认 Python 路径写死。
   - `application.yml` 默认指向某个用户的 Conda 路径，换机器或云服务器会失败。

3. 实验对比是 Mock。
   - 页面上虽然有实验对比，但 `runExperiments()` 仍走前端模拟。

4. 反向传播说明区是占位。
   - 页面明确写了“逐层反向传播计算图、梯度热力图和真实权重更新尚未实现”。

5. 自定义上传数据集不能真实训练。
   - 后端导入了 metadata/preview，但 Python worker `load_dataset()` 不支持 `upload-*`。

6. 数据集名称与真实数量容易不一致。
   - `mnist-1000` 文案写“全量 70000”，命名不合理。

### 3.3 B 负责人后续任务

#### B-P0：保证真实训练可跑通

必须完成：

- 执行并验证：

```powershell
cd backend/spring
python scripts/download_builtin_datasets.py
```

- 至少验证 3 个任务：
  - `iris`：最快，必须跑通。
  - `points-2d`：快，适合演示。
  - `mnist-1000` 或小 MNIST：图像任务，必须至少一个跑通。

- 修正 `application.yml`：
  - 默认 Python 不要写死个人路径。
  - 建议改为 `${DEEPVISION_TRAINING_PYTHON:python}`。
  - 在部署文档说明 Windows/Conda/Linux 如何配置。

- 增加启动前检查：
  - 数据集目录是否存在。
  - Python worker 脚本是否存在。
  - Python 是否能 import torch/PIL。

#### B-P0：整理训练页面，隐藏 Mock 暴露

必须完成：

- 把 B 页面里“实验对比（前端辅助模拟）”改为“快速理论预测”或移到次要区域。
- 把“反向传播说明（教学占位）”改为：
  - 若交给 D：写“逐层反向传播请查看 D 模式”。
  - 不要写“尚未实现”这种扣分文案。
- 帮助文档 `help-manual.component.ts` 仍写“Training is currently simulated”，需要改掉，否则会和真实 B 后端冲突。

#### B-P1：真实结构影响实验

目标：拿“学习引导与评估系统”满分。

实现建议：

- 新增 `experimentJobs` 状态。
- 提供三个按钮或一个“一键真实对比”：
  - Baseline：当前网络。
  - Deeper：多加一个 Dense 或 Conv。
  - Activation：ReLU 换 Tanh/GELU。
  - Optimizer：SGD 换 Adam。
- 每个实验用小数据集、少 epoch 串行训练。
- 输出对比：
  - final test accuracy
  - final val accuracy
  - final loss
  - elapsed seconds
  - best epoch

注意：

- 不要和普通训练 job 抢同一个 `TrainingRuntimeService` 状态。
- 可以后端新增 `/api/training/experiments/start`，也可以前端串行调用现有 `/api/training/start`。

#### B-P1：B 模式内部的训练后单样本推理

目标：补齐“训练后单样本推理演示”。

建议新增：

- Spring：
  - `POST /api/training/checkpoints/{checkpointId}/forward`
- Python worker：
  - `action=forward_checkpoint`
  - 加载 checkpoint。
  - 读取 dataset 中指定样本或上传样本。
  - 用 forward hook 收集中间层输出。
  - 返回 layerActivations、shape、featureMap previews、prediction。

展示位置：

- 只在 B 页面内部展示，不跳转 A，不向 A/C/D 传递 checkpoint。
- 可放在测试结果样本卡片旁边：
  - “查看该样本的训练后推理过程”
  - 展开后显示 Conv feature maps、Dense activations、最终概率。

这样既满足评分项中的“训练后单样本推理”，又不破坏四人独立开发边界。

#### B-P2：真实权重直方图

当前是根据均值方差生成近似图。可优化为：

- Python worker 每个 epoch 返回权重分位数/直方图 bins。
- 或后端读取 checkpoint 生成真实直方图。

不是最高优先级，但答辩追问时很有帮助。

## 4. C 模式：卷积过程、CNN 解释、Grad-CAM

### 4.1 当前 C 模式真实功能

C 当前不是简单占位。它有这些真实能力：

- 已经不是 iframe 宿主，而是 Angular 原生组件。
- 使用 TensorFlow.js 加载静态模型 `/mode-c/cnn-explainer/assets/data/model.json`。
- 使用静态图片样本 espresso/panda/pizza/bus。
- 在浏览器中做真实 TF.js 推理。
- 能生成中间层 summaries、previews、layerDetails。
- 对卷积层能构造 patch、kernel、products、weighted sum、bias、outputValue 等解释数据。
- 有 Conv/ReLU/Pool/Softmax 相关联动面板。

### 4.2 C 模式存在的问题

1. 不是 B 训练模型的解释。
   - 它解释的是 cnn-explainer 静态模型。
   - 这符合模式独立约束，不需要改成解释 B checkpoint。

2. 没有真实 Grad-CAM。
   - 当前没有 Grad-CAM hook、梯度回传、热力图叠加。
   - `SimEngine.refreshVisuals()` 有 `gradCamMap`，但这是模拟数据，且主要属于 A/B 旧逻辑，不是 C 的真实可解释性。

3. 部分文案仍显示“in-progress/planned/fallback”。
   - 这些开发状态信息不适合答辩界面。

4. 中文乱码很严重。
   - C assets 中大量中文乱码，必须修。

### 4.3 C 负责人后续任务

#### C-P0：修复 C 文案和页面状态

必须完成：

- 修复 `mode-c-assets.service.ts` 中 sample title、description、overviewStages、detailTopics、milestones、articleSections 的乱码。
- 删除或改写“占位”“planned”“in-progress”暴露给用户的文案。
- 把 C 明确定位为“CNN 卷积过程与中间特征解释”。

#### C-P0：完善卷积过程教学闭环

确保答辩能演示：

1. 选择样本图片。
2. 显示预测 top classes。
3. 选择 Conv 层。
4. 显示输入局部 patch。
5. 显示卷积核权重。
6. 显示逐元素乘积。
7. 显示加权求和 + bias。
8. 显示输出 feature map。
9. 切换通道并解释不同通道捕捉不同特征。

#### C-P1：实现 Grad-CAM

这是 C 最适合补的进阶功能。

可选实现路线：

路线 A：前端 TF.js 实现，和当前 C 最贴合。

- 找到最后一个 conv 层。
- 选择目标类别 score。
- 用 `tf.grad` 或 `tf.variableGrads` 计算目标类别对 conv activation 的梯度。
- 对梯度做 global average pooling 得到每个通道权重。
- 加权求和 conv feature maps。
- ReLU、normalize、resize 到原图。
- 叠加 heatmap。

建议：

- C 做路线 A，确保独立可演示。
- 文案说明：“C 模式使用内置 CNN Explainer 模型做 Grad-CAM 教学演示。”

#### C-P1：C 模式内部的特征图与 Grad-CAM 报告

不要把 C 的特征图导出给 A/B。C 自己完成一份可演示的解释报告即可：

- 当前样本。
- Top-5 类别概率。
- 选中卷积层的通道特征图。
- 选中通道的 patch/kernel/product 计算过程。
- Grad-CAM 热力图叠加。
- 一段自动生成的文字解释。

这样 C 可以独立拿“卷积过程”“特征图可视化”“Grad-CAM 可解释性”分数。

## 5. D 模式：MLP、反向传播、优化器

### 5.1 当前 D 模式真实功能

D 是前端本地教学模式，当前已有：

- TypeScript 实现 MLP forward pass。
- CrossEntropy/MSE loss。
- 反向传播计算每层梯度。
- 参数更新前后快照。
- SGD、Momentum、Adam。
- XOR、Spiral、Circle、Blob 数据集生成。
- step-by-step animation：forward、loss、backward、update。
- 神经元、边权重、梯度、bias 细节。

### 5.2 D 模式存在的问题

1. 没有后端。
   - 这不是问题，但答辩时必须定位成“教学解释引擎”，不要说是真实大规模训练。

2. 决策边界功能可能未完全接到 UI。
   - engine 有 `computeDecisionBoundary()`，但 state 中 `decisionBoundary` 没看到定期更新逻辑。

3. 中文乱码。
   - D assets 文案乱码非常多。

4. 没有 AI 助手。
   - D 很适合接 AI 解释反向传播，但目前没挂。

### 5.3 D 负责人后续任务

#### D-P0：修复 D 文案和定位

必须完成：

- 修复 `mode-d-assets.service.ts` 和 D 页面文案乱码。
- 页面上明确写成：
  - “MLP 反向传播教学”
  - “小型二维数据集”
  - “逐步观察梯度与参数更新”
- 不要使用“真实训练后端”之类表述。

#### D-P0：把反向传播讲完整

确保答辩流程：

1. 选择 XOR MLP。
2. 单步执行。
3. 展示 forward 激活。
4. 展示 loss。
5. 展示 backward 梯度从输出层回传到隐藏层。
6. 展示权重更新前后变化。
7. 切换 SGD/Momentum/Adam 对比收敛。

#### D-P1：补决策边界可视化

如果 UI 已有但未更新，修通它；如果没有，新增：

- 每 N 次训练后计算一次 decision boundary。
- 在二维数据点背景上画分类区域。
- 支持切换网络结构和优化器后观察边界变化。

这能强力支撑“结构影响实验”和“学习引导”。

#### D-P1：优化器对比教学

做成无需后端的独立实验：

- 同一数据集、同一网络、同一初始权重。
- 分别跑 SGD、Momentum、Adam。
- 显示 loss 曲线、收敛速度、决策边界。

注意：

- 若要公平对比，必须固定随机种子或复用初始权重。

#### D-P2：接入 AI 助手

D 的 AI contextProvider 应包含：

- 当前数据集。
- 当前网络结构。
- 当前 loss。
- 当前优化器。
- 当前选中神经元。
- 当前梯度范数、权重变化。

预设问题：

- “解释当前这条边的梯度为什么是正/负。”
- “为什么 Adam 比 SGD 收敛更快。”
- “当前学习率是否过大。”

## 6. 全局任务

这些任务不属于某个模式，但对满分非常关键。

### 6.1 文档与乱码修复

优先级：P0。

负责人：建议由当前任务较轻的人统一负责，所有成员配合检查自己模式。

必须产出：

- `README.md`：项目介绍、架构、启动方式、功能入口。
- `docs/需求分析.md`：课程要求映射、用户角色、使用场景。
- `docs/系统设计.md`：前后端架构、模块划分、数据流、WebSocket 流程、训练 worker 流程。
- `docs/接口说明.md`：Auth、Forward、Training、LLM、Checkpoint 接口。
- `docs/部署说明.md`：本地部署、Docker 部署、云部署。
- `docs/用户手册.md`：ABCD 四模式使用流程。
- `docs/小组分工.md`：四人分工、贡献说明。
- `AI_USAGE.md`：AI 工具使用记录，修复乱码。

必须修复：

- 首页乱码。
- B/C/D 大量乱码。
- README 乱码。
- Spring 后端 README 乱码。
- help manual 与当前真实功能不一致的内容。

### 6.2 LLM 全局接入

优先级：P1。

当前：

- 后端已经有 `/api/llm/chat`。
- A 已经接入。
- 需要 `ARK_API_KEY`。

建议任务：

- 增加 `LlmContextProvider` 给 B/C/D。
- B：训练调参助手。
- C：CNN/Grad-CAM 解释助手。
- D：反向传播/优化器解释助手。
- 增加未配置 API Key 的友好提示。
- 增加 `.env.example` 或部署文档中写：

```powershell
$env:ARK_API_KEY="..."
$env:ARK_MODEL="doubao-seed-2-0-pro-260215"
```

### 6.3 WebSocket 协作讨论

优先级：P2。

当前：

- B 的 WebSocket 只广播训练指标。
- 多个用户打开同一个 jobId 可以观察同一训练过程。
- 没有在线讨论。

建议：

- 在 Spring 新增 `/api/collab/stream?roomId=...`。
- 或复用训练 WebSocket，支持客户端发送 `{type:"chat", roomId, user, text}`。
- B 页面右侧增加“训练讨论”面板。
- 只做内存房间即可，不必持久化。

这样可以明确满足“实时协作：多个用户同时观察同一模型训练过程，并在线讨论”。

### 6.4 Docker 与部署

优先级：最后做，但必须做。

为什么最后：

- 当前功能、数据、环境变量还在变。
- 先部署会反复改镜像，浪费时间。

必须产出：

- `frontend/Dockerfile`
- `backend/spring/Dockerfile`
- `backend/python-forward/Dockerfile`
- `backend/python-training` 可并入 Spring 镜像或单独镜像。
- `docker-compose.yml`
- `nginx.conf`
- `.env.example`
- `docs/部署说明.md`

推荐部署架构：

- Nginx：
  - 80/443 对外。
  - `/` -> Angular 静态文件。
  - `/api` -> Spring 8080。
  - `/api/training/stream` -> Spring WebSocket。
- Spring：
  - 8080。
  - 挂载 uploads、datasets、training-jobs、H2 data。
- Python forward：
  - 5000。
- Python training：
  - 简化做法：Spring 镜像里安装 Python + torch + worker 脚本。
  - 更干净做法：训练服务单独容器，Spring 通过 HTTP 调用，但改造量较大。

部署前必须：

- 先下载数据集，或在容器启动时执行数据集初始化。
- 注意 MNIST/CIFAR 全量会让镜像/卷很大。建议准备 small 数据集用于演示。

### 6.5 测试与演示脚本

优先级：P1。

建议补最小测试：

- Spring：
  - AuthController 注册/登录测试。
  - TrainingDatasetService 内置列表测试。
  - TrainingJobService split 校验测试。
  - LlmController 未配置 key 错误测试。
- Python：
  - `training_worker.py` 用 Iris/points 小数据跑 1 epoch 的 smoke test。
  - `forward_engine.py` Conv/Pool/Dense shape test。
- 前端：
  - 至少 build 通过。
  - 可写 Playwright e2e：访问 A/B/C/D 页面，确认关键文本和按钮存在。

答辩脚本：

- `scripts/start-all-dev.ps1`
- `scripts/check-health.ps1`
- `scripts/prepare-demo-data.ps1`

## 7. 四人不冲突分工建议

### 成员 A：A 模式负责人

主线：前向传播、卷积核影响、单样本推理展示。

已完成或基本完成：

- 已完成：删除 A 的 Mock 训练区和训练状态残留。
- 已完成：补充层公式解释和 Dense/Output 权重来源说明。
- 已完成：补充卷积核对比弹窗，展示多种经典 kernel 的真实 forward 结果。
- 已基本完成：A 内部单样本前向传播展示，包含每层输出、特征图/向量、层检查器和历史记录。
- 已基本完成：基础网络编辑器能力，支持添加、删除、重排层并自动形成线性连接。

剩余建议：

- A-P0 最后检查 A 页面和首页入口是否还有乱码或“训练/Mock”残留文案。
- A-P0 固化 A 的答辩演示脚本，确保 forward 后端未启动时错误提示清楚。
- A-P1 增加 A 模式 AI 助手预设问题，重点解释卷积核、shape 公式、Dense 权重来源、当前层输出和配置错误。
- A-P1 可以补一段 A 模式使用说明或截图到文档，证明 A 是真实 forward，不是模拟。

不要做：

- 不要在 A 中继续做真实训练，避免和 B 冲突。
- 不要在 A 中做 LSTM；LSTM 更适合 D 的序列/门控教学可视化。
- 不要在 A 中做 ResNet/Residual Add；若团队要做完整 ResNet，交给 B。
- 不要继续投入任意连边/DAG 编辑器，除非 B 的 ResNet 明确需要共享组件。

### 成员 B：B 模式负责人

主线：真实训练、数据集、评估、训练实验。

已完成或基本完成：

- 已有 Spring 后端训练任务、Python PyTorch worker 和 WebSocket 指标流。
- 已有训练控制、optimizer/scheduler 配置、测试集评估、checkpoint 和训练曲线。

剩余建议：

- B-P0 跑通真实数据集下载和训练。
- B-P0 修复 Python 路径和环境检查。
- B-P0 清理 B 页面中的 Mock/占位暴露。
- B-P1 做真实结构影响实验。
- B-P1 做 B 内部 checkpoint 单样本 forward 后端和页面展示。
- B-P1 确保 checkpoint 保存/测试稳定。
- B-P2 承担 ResNet 进阶功能：优先做小型 ResNet/Residual CNN 训练模板和测试集评估；如果时间不足，至少做固定 residual block 模板，不做通用 DAG 编辑器。
- B-P2 可选做真实权重直方图。
- B-P2 若要做实时协作，建议只围绕 B 的训练 job 做 WebSocket 聊天房间，不做全局聊天室。

不要做：

- 不要在 B 里重新做 D 的逐层反向传播教学。
- B 只提供真实训练指标和必要接口。

### 成员 C：C 模式负责人

主线：卷积过程、CNN 中间层解释、Grad-CAM。

已完成或基本完成：

- 已有 TF.js 静态模型推理、中间层 summary、特征图、卷积 patch/kernel/product 解释。

剩余建议：

- C-P0 修复 C 文案乱码和开发状态暴露。
- C-P0 完整打磨卷积过程解释闭环。
- C-P1 实现 Grad-CAM。
- C-P1 补 feature map/通道解释答辩路径。
- C-P2 可选接入 AI 解释 CNN/Grad-CAM；若接口未稳定，可先不做。

不要做：

- 不要依赖 B checkpoint。
- 不要向 A/B 输出业务数据。
- C 独立使用静态 TF.js CNN 模型拿解释性分。

### 成员 D：D 模式负责人

主线：MLP、反向传播、优化器、决策边界。

已完成或基本完成：

- 已有本地教学引擎，能展示 MLP 前向、loss、反向传播、SGD/Momentum/Adam 更新。

剩余建议：

- D-P0 修复 D 文案乱码。
- D-P0 打磨逐步反向传播流程。
- D-P1 补决策边界可视化。
- D-P1 做优化器对比教学。
- D-P2 可选做 LSTM Cell 教学：展示 input/forget/output gate、candidate、cell state、hidden state 随时间步变化，用于补“复杂结构”进阶展示。
- D-P2 可选接入 AI 解释反向传播；若接口未稳定，可先不做。

额外适合承担：

- 如果 D 功能较稳定，D 成员适合承担全局文档初稿或 Docker 方案草稿。

不要做：

- 不要把 D 改成后端真实训练系统；D 的价值是教学可解释性。

### 全局负责人或轮值

建议由“进度最快的人 + 组长”承担：

- 文档整合。
- 乱码统一修复检查。
- Docker 和部署。
- 演示脚本。
- 最终答辩 PPT。

## 8. 推荐实施顺序

### 第 1 阶段：止血与可演示

目标：所有模式页面不自曝短板，核心链路能跑。

1. 修复 README、首页、B/C/D 文案乱码。
2. 执行数据集下载脚本，跑通 B 的 Iris/points/MNIST 至少三个任务。
3. 修改 B 帮助文档和页面占位文案。
4. A 固化前向传播演示。
5. C 固化卷积过程演示。
6. D 固化反向传播单步演示。

### 第 2 阶段：补评分硬缺口

目标：基础功能拿满。

1. B 真实结构影响实验。
2. B 模式内部 checkpoint 单样本推理展示。
3. C 特征图解释完善。
4. 全局 LLM 接入 B/C/D。
5. 文档补齐需求、设计、接口、用户手册。

### 第 3 阶段：进阶亮点

目标：冲进阶和创新分。

1. C 实现 Grad-CAM。
2. D 实现/打磨决策边界与优化器对比。
3. B 实现 ResNet/Residual CNN 训练模板或固定残差块训练预设。
4. D 可选实现 LSTM Cell 教学可视化。
5. B 可选实现围绕训练 job 的 WebSocket 协作讨论。

### 第 4 阶段：部署与答辩

目标：工程能力拿满。

1. 写 Dockerfile 和 docker-compose。
2. Nginx 反代，支持 WebSocket。
3. 公有云部署。
4. 准备 `.env.example`。
5. 准备演示脚本和故障预案。
6. 最终跑一遍完整答辩流程。

## 9. 满分答辩推荐故事线

不要按“评分表散点”讲，按学习路径讲：

1. **A 模式：模型如何看一张图**
   - 展示图片输入、卷积核、逐层前向传播、特征图、3D 网络。

2. **B 模式：模型如何被训练出来**
   - 选择数据集，配置网络和超参数，真实 PyTorch 训练，WebSocket 实时曲线，测试集评估，checkpoint。

3. **C 模式：CNN 为什么这样判断**
   - 展示卷积局部计算、中间特征图、Grad-CAM 热力图。

4. **D 模式：参数为什么会更新**
   - 展示 MLP 的 loss、梯度回传、权重变化、优化器对比、决策边界。

5. **全局能力**
   - 登录、记录、A 模式 LLM 助手、B 模式 WebSocket 指标流/可选协作、Docker/云部署、文档和测试。

## 10. 最后检查清单

答辩前必须逐项确认：

- [ ] 首页和 ABCD 页面没有乱码。
- [ ] README 能按步骤启动项目。
- [ ] `npm run build` 成功。
- [ ] `mvn test` 成功。
- [ ] Python forward 服务启动成功。
- [ ] Spring 启动成功。
- [ ] B 数据集目录存在。
- [ ] B 至少一个图像任务真实训练完成。
- [ ] B 至少一个表格/二维任务真实训练完成。
- [ ] A 前向传播能展示每层输出。
- [ ] A 不再暴露 Mock Training 或训练控制残留。
- [ ] A 层公式解释卡片可正常展示。
- [ ] A 卷积核对比面板可调用真实 forward 后端并显示结果。
- [ ] A AI 助手有面向卷积核、shape、公式、当前层输出的预设问题。
- [ ] C 能展示卷积过程和特征图。
- [ ] D 能单步展示反向传播。
- [ ] LLM 配置 `ARK_API_KEY` 后可用；未配置时提示清楚。
- [ ] Docker compose 可启动，或至少部署文档完整。
- [ ] 公有云地址可访问。
- [ ] 演示脚本不依赖现场临时下载大文件。


写各自的开发文档
C和D的feature换个位置。
