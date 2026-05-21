# DeepVision Studio 满分冲刺缺口与分工方案

> 更新时间：2026-05-21  
> 范围：根据当前仓库代码重新核对完成度，只重写本计划文档，不改动业务代码。  
> 结论：当前代码已经覆盖 A/B/C/D 四个核心模式，并额外补了 Mode E、Docker、训练聊天室、B/C/D 的 LLM 接入、C 的 Grad-CAM、D 的决策边界等旧计划中的多项缺口；但距离“所有任务要求全部完成”还有关键风险，尤其是内置数据集未落地、B 的训练后单样本逐层激活未实现、自动化测试不足、云部署与答辩材料未闭环。

## 1. 当前总体完成度

### 已基本完成

- 前端 Angular 路由完整：`/mode-a`、`/mode-b`、`/mode-c`、`/mode-d`、`/mode-e`、`/network-3d`、`/training/experiments`、`/training/collaboration`。
- Spring 后端已覆盖认证、JWT、A 前向传播代理、A 记录、B 训练任务、训练 WebSocket、checkpoint、训练聊天室、LLM 代理。
- Python forward 服务负责 A 的真实前向传播计算。
- Python training worker 负责 B 的 PyTorch 训练和 checkpoint 测试。
- Docker 相关文件已存在：`docker-compose.yml`、`frontend/Dockerfile`、`frontend/nginx.conf`、`backend/spring/Dockerfile`、`backend/python-forward/Dockerfile`。
- LLM 浮动助手已接入 A/B/C/D，不再是旧计划中的“A-only”状态。
- C 模式已经是 Angular 原生 TF.js CNN Explainer，不是 iframe 宿主；并已实现前端 Grad-CAM 计算和热力图叠加。
- D 模式已经有 MLP 前向、loss、反向传播、SGD/Momentum/Adam、决策边界、优化器对比曲线和 LLM 助手。
- B 模式已经有训练聊天室 WebSocket，并支持 `@智能助手` 调用 LLM 结合训练状态回答。
- B 已有 checkpoint 历史对比页，可基于真实训练历史做结构与结果对比。

### 尚未全部完成

- 仓库中未发现 `backend/spring/datasets` 真实内置数据集目录；B 真实训练仍依赖答辩前下载或挂载数据。
- `application.yml` 的 `DEEPVISION_TRAINING_PYTHON` 默认值仍是个人 Conda 路径，Docker 中有覆盖，但本地换机启动仍容易失败。
- B 的 `/training/experiments` 当前更像“checkpoint 历史对比”，不是“一键派生多个结构并自动串行真实训练”的实验系统。
- B checkpoint 只支持重新跑测试集；还没有“选择单个样本并返回训练后模型逐层激活/feature map”的接口。
- Spring 没有 `src/test`，前端只有默认 `app.spec.ts`，自动化测试覆盖不足。
- Docker compose 已有，但未确认当前环境下一键启动、数据初始化和训练链路全部跑通。
- 未看到公有云部署地址、HTTPS/HTTP2 配置和完整答辩演示脚本。

## 2. 评分项完成情况对照

| 评分项 | 当前状态 | 结论 |
| --- | --- | --- |
| 网络结构编辑 | A/B 支持线性层编辑、拖拽、删除、重排、参数配置；B 代码中已有 residual 类型展示能力 | 基础够用，通用 DAG/任意连边仍不是重点 |
| 单样本前向传播 | A 已能调用真实 forward 后端展示逐层输出；C 可做 TF.js 样本推理 | A/C 完成教学版；B 训练后单样本逐层激活仍缺 |
| 真实训练 | B 已有 Spring + PyTorch worker + WebSocket 指标 | 代码具备，但必须补数据和环境验证 |
| 数据集管理 | B 有内置数据集注册和上传逻辑 | 数据文件未落地是最大风险；上传数据集不等于可被 worker 真实训练 |
| 训练过程可视化 | B 有 loss/accuracy/lr/梯度/权重统计和日志 | 训练曲线完成；真实逐层权重直方图和训练中 feature map 仍有限 |
| checkpoint | B 可保存、列出、重新测试 checkpoint | 完成基础；缺 checkpoint 单样本 forward |
| 结构影响实验 | B 有 checkpoint 历史对比页；D 有教学级优化器/结构对比 | 已能讲，但不是自动派生真实实验；满分建议继续补 |
| CNN 解释 | C 有 TF.js 推理、中间层、卷积 patch/kernel/product | 完成度高 |
| Grad-CAM | C 已实现 `computeGradCam()`，生成 heatmap 和 overlay | 旧计划缺口已完成，答辩前需实际跑浏览器验证 |
| 反向传播教学 | D 有 MLP 反传、参数更新、优化器 | 完成度高 |
| 决策边界 | D 已有 `computeDecisionBoundary()` 并接到 UI 图表 | 旧计划缺口已完成 |
| LLM 辅助 | A/B/C/D 已接入浮动助手；B 聊天室也接 LLM | 旧计划缺口已完成，需验证未配置 key 时提示 |
| 实时协作 | B 已有训练聊天室、在线成员、历史消息、智能助手 | 旧计划缺口已基本完成 |
| Docker 部署 | compose、Dockerfile、Nginx 配置已存在 | 代码层完成，仍需实际验证和文档闭环 |
| 测试 | 只有默认前端 spec，无后端测试目录 | 明显缺口 |

## 3. 关键风险清单

### P0：数据集没有落地

当前 `backend/spring/datasets` 不存在或为空。B worker 依赖 `DEEPVISION_DATASET_ROOT` 下的内置数据，训练演示会直接受影响。

必须完成：

```powershell
cd backend/spring
python scripts/download_builtin_datasets.py
```

至少验证：

- `iris` 能完成 1-3 epoch。
- `points-2d` 能完成 1-3 epoch。
- `mnist-1000` 或等价 small 图像任务能完成一次训练、测试和 checkpoint 保存。

### P0：本地 Python 路径默认值不通用

`backend/spring/src/main/resources/application.yml` 仍默认：

```yaml
python-executable: ${DEEPVISION_TRAINING_PYTHON:C:/Users/lizihan/miniconda3/envs/dl-platform/python.exe}
```

Docker 中已用 `DEEPVISION_TRAINING_PYTHON=python3` 覆盖，但本地答辩机未设置环境变量时仍会失败。

建议改法后续由 B/部署负责人执行：

```yaml
python-executable: ${DEEPVISION_TRAINING_PYTHON:python}
```

并在部署文档写清楚 Windows Conda、系统 Python、Docker 三种配置。

### P0：B 缺“训练后单样本逐层激活”

评分中最容易被追问的是：“训练完成后输入单个样本，能否看到该训练后模型每层激活？”

当前状态：

- A 可以展示教学版真实 forward，但不是 B 训练后的模型。
- B 可以 checkpoint 测试集重评估，但没有 checkpoint 单样本 forward + hooks。

建议由 B 补：

- `POST /api/training/checkpoints/{checkpointId}/forward`
- Python worker 增加 `action=forward_checkpoint`
- 加载 checkpoint，选择 dataset 中一个样本或上传样本。
- 用 PyTorch forward hook 收集 Conv feature maps / Dense activations。
- B 页面在测试样本卡片旁展示预测结果、概率、逐层 shape、feature maps。

### P1：真实结构影响实验还不够自动化

当前 `/training/experiments` 读取 checkpoint 历史做对比，这是真实记录，不是 Mock；但它不是旧计划中设想的“一键派生多个实验并自动训练”。

建议后续两种取舍：

- 保守方案：把它定位为“真实训练历史对比”，要求答辩前手动准备 baseline/deeper/optimizer 三个 checkpoint。
- 满分方案：新增“一键真实对比”，自动串行启动 2-3 个小 epoch 训练任务并生成对比表。

### P1：自动化测试不足

当前没有后端测试目录，前端只有默认 spec。建议补最小测试，不追求大覆盖：

- Spring：Auth 注册登录、dataset 列表、checkpoint 权限、LLM 未配置 key。
- Python：`forward_engine.py` shape 测试；`training_worker.py` 用 Iris/points 跑 1 epoch smoke test。
- 前端：至少 build；可加简单页面 smoke e2e。

### P1：Docker 与部署尚需实测

compose 已存在，但还要确认：

- `docker compose up --build` 可以启动三项服务。
- Nginx 能代理 `/api`、WebSocket、uploads/datasets。
- Spring 容器内 Python training worker 能 import torch/PIL。
- 数据集卷已初始化。
- 训练任务在容器内能完成。

### P2：公有云、HTTPS/HTTP2、答辩材料

这些属于工程展示加分项：

- 公有云地址。
- HTTPS 或至少 Nginx 反代说明。
- `.env.example`。
- `scripts/start-all-dev.ps1`、`scripts/check-health.ps1`、`scripts/prepare-demo-data.ps1`。
- 用户手册、接口说明、系统设计、部署说明、小组分工、AI 使用记录。

## 4. A 模式后续计划

### 当前定位

A 是“前向传播与卷积核影响教学实验室”。它不承担真实训练，避免和 B 冲突。

### 已完成

- 图片样本选择和上传。
- 网络层编辑、参数配置、线性连接展示。
- Spring -> Python forward 真实计算。
- 每层输出 shape、特征图/向量、最终 tensor。
- 层公式解释。
- Dense/Output 演示权重来源说明。
- 卷积核对比弹窗。
- A 记录保存/恢复。
- 3D 网络查看器入口。
- LLM 浮动助手。
- 旧训练 Mock 区域已清理。

### 剩余任务

- P0：最后检查 A 页面、首页入口和帮助文档，不要出现“训练 Mock”“模拟训练”等误导文案。
- P0：固化演示脚本，确认 forward 服务未启动时错误提示清楚。
- P1：把 A 的使用流程和截图写入用户手册或 A 开发文档。

### 不建议继续做

- 不在 A 中做真实训练。
- 不在 A 中接 B checkpoint。
- 不在 A 中扩展通用 DAG/任意连边。
- 不在 A 中做 LSTM；序列门控教学更适合 D 或独立 Mode E。

## 5. B 模式后续计划

### 当前定位

B 是“真实训练、评估、checkpoint、训练协作实验室”。

### 已完成

- Spring 训练任务管理。
- Python PyTorch training worker。
- WebSocket 训练指标流。
- 开始、暂停、恢复、停止、重置。
- optimizer/scheduler/epoch/batch/lr 等配置。
- 测试集评估和样本预测结果。
- checkpoint 保存、列表、重新测试。
- 权重统计、训练日志、曲线展示。
- LLM 浮动助手。
- 训练聊天室、在线成员、历史消息、`@智能助手`。
- checkpoint 历史对比页。

### 剩余任务

- P0：下载并验证内置数据集。
- P0：修复本地 Python 默认路径，或至少在文档和 `.env.example` 中强制说明。
- P0：确认训练页面无“占位/未实现/模拟训练”自曝文案。
- P1：补 checkpoint 单样本 forward + 逐层激活。
- P1：把实验对比页明确为“真实 checkpoint 历史对比”，或继续实现“一键真实结构实验”。
- P1：准备 baseline/deeper/optimizer 三组可展示 checkpoint。
- P2：可选补真实权重直方图 bins。
- P2：可选补小型 ResNet/Residual CNN 训练预设。

### 不建议继续做

- 不在 B 中重复 D 的逐层反向传播教学。
- 不把 B checkpoint 传给 A/C 做跨模式联动。

## 6. C 模式后续计划

### 当前定位

C 是“CNN 卷积过程、中间特征、Grad-CAM 可解释性实验室”。

### 已完成

- Angular 原生组件，不再依赖 iframe 宿主。
- 加载 TF.js 静态 CNN Explainer 模型。
- 内置样本图片推理。
- Top classes 预测结果。
- 中间层 summary、preview、detail。
- 卷积 patch、kernel、逐元素乘积、bias、输出值解释。
- ReLU/Pool/Softmax 相关解释面板。
- Grad-CAM：计算目标类别对最后卷积层的梯度，生成 heatmap、overlay 和 dominant channels。
- LLM 浮动助手。

### 剩余任务

- P0：浏览器实测 C 页面，确认 TF.js 模型、样本图、Grad-CAM heatmap 都能正常渲染。
- P0：检查 C 文案中是否还有 planned、fallback、in-progress 或乱码。
- P1：完善 C 的解释报告：当前样本、Top-5、选中通道、卷积计算、Grad-CAM 叠加、自动文字解释。
- P1：把 C 的使用流程写入开发文档和用户手册。

### 不建议继续做

- 不依赖 B checkpoint。
- 不向 A/B 输出特征图业务数据。
- 不把 C 改成训练模块。

## 7. D 模式后续计划

### 当前定位

D 是“MLP、反向传播、优化器、决策边界教学实验室”。

### 已完成

- TypeScript 本地 MLP 引擎。
- forward、loss、backward、update 四阶段。
- CrossEntropy/MSE。
- SGD、Momentum、Adam。
- XOR、Spiral、Circle、Blob 数据集。
- 权重、bias、梯度、参数更新前后快照。
- 决策边界计算并接入 UI 图表。
- 优化器对比曲线。
- LLM 浮动助手。

### 剩余任务

- P0：浏览器实测 D 的单步流程、连续训练、决策边界、优化器对比。
- P0：检查 D 文案，不要出现乱码或“真实后端训练”误导。
- P1：把 D 的答辩流程写入开发文档和用户手册。
- P2：如果需要“复杂结构”加分，可由 D 或 Mode E 承担 LSTM/Transformer Cell 教学；不要挤占 D 的反传主线。

### 不建议继续做

- 不把 D 改成后端真实训练系统。
- 不和 B 的 PyTorch 训练链路联动。

## 8. Mode E 说明

当前路由中已经存在 `/mode-e`，代码文件包括 `mode-e-transformer.engine.ts` 和 `mode-e-page.component.*`。这不是旧 ABCD 计划的一部分，但可以作为额外进阶展示。

建议定位：

- 如果 Mode E 已可稳定演示，就作为“Transformer/注意力机制”扩展亮点。
- 如果未稳定，不要让它进入主答辩链路，只在文档中写成可选探索模块。

主答辩仍建议围绕 A/B/C/D 四条学习路径组织。

## 9. 全局任务

### P0：可运行闭环

- 下载数据集。
- 跑通 Spring。
- 跑通 Python forward。
- 跑通 B 至少三个小任务。
- 跑通 A/C/D 页面核心演示。
- 跑通 LLM 未配置和已配置两种状态。

### P1：文档闭环

建议补齐或更新：

- `README.md`：项目介绍、架构、启动方式、功能入口。
- `DOCKER.md` 或 `docs/部署说明.md`：本地、Docker、云部署。
- `docs/需求分析.md`：课程要求映射。
- `docs/系统设计.md`：前后端架构、数据流、WebSocket、训练 worker。
- `docs/接口说明.md`：Auth、Forward、Training、Checkpoint、LLM、Collaboration。
- `docs/用户手册.md`：A/B/C/D 使用流程。
- `docs/小组分工.md`：四人贡献和边界。
- `AI_USAGE.md`：AI 工具使用记录。
- 各自开发文档：A/B/C/D 每人一份，写清楚负责功能、核心实现、演示步骤、已知限制。

### P1：测试和脚本

建议新增：

- `scripts/prepare-demo-data.ps1`
- `scripts/start-all-dev.ps1`
- `scripts/check-health.ps1`
- Python smoke test。
- Spring 最小单元/集成测试。
- 前端 build 和页面 smoke test。

### P2：部署展示

- 验证 `docker compose up --build`。
- 准备 `.env.example`。
- 云服务器部署。
- Nginx 反代 WebSocket。
- 记录公网地址和故障预案。

## 10. 推荐实施顺序

### 第 1 阶段：先保证能演示

1. 下载并固定 B 数据集。
2. 修正或文档化 Python 路径。
3. 本地完整启动 frontend、Spring、python-forward。
4. B 跑通 Iris、points、MNIST small。
5. A/C/D 各自跑一遍核心演示。
6. 清理页面上的乱码、占位、planned、fallback、未实现等文案。

### 第 2 阶段：补硬评分点

1. B checkpoint 单样本 forward + 逐层激活。
2. B 准备真实结构对比数据，优先用 checkpoint 历史页展示。
3. C 实测并打磨 Grad-CAM 报告。
4. D 实测并打磨决策边界和优化器对比。
5. B 训练聊天室实测多人连接和 `@智能助手`。

### 第 3 阶段：工程化和材料

1. Docker compose 实测。
2. 补 `.env.example`。
3. 补测试和 health 脚本。
4. 补需求、设计、接口、用户手册、部署、小组分工、AI 使用记录。
5. 每个成员写自己的开发文档。

### 第 4 阶段：答辩彩排

1. 按固定脚本完整走一遍。
2. 准备好 checkpoint 和小数据集，避免现场下载大文件。
3. 准备 forward、training、LLM、Docker 四类故障预案。
4. 云地址可访问时再加入答辩材料；不可访问则明确展示 Docker 和本地部署。

## 11. 答辩故事线

1. **A：模型如何看一张图**
   - 图片输入、卷积核、逐层 forward、特征图、公式解释、3D 网络。

2. **B：模型如何被训练出来**
   - 数据集、网络结构、超参数、真实 PyTorch 训练、WebSocket 曲线、测试集评估、checkpoint、训练聊天室。

3. **C：CNN 为什么这样判断**
   - TF.js 推理、卷积局部计算、中间特征图、Grad-CAM 热力图。

4. **D：参数为什么会更新**
   - MLP forward、loss、反向传播、权重更新、优化器对比、决策边界。

5. **全局工程能力**
   - 登录、记录、LLM 助手、WebSocket 协作、Docker、文档、测试、部署。

## 12. 最后检查清单

- [ ] `backend/spring/datasets` 已有演示数据。
- [ ] `iris`、`points-2d`、`mnist-1000` 或 small 图像任务真实训练通过。
- [ ] `DEEPVISION_TRAINING_PYTHON` 在本机和 Docker 中都配置正确。
- [ ] A 前向传播每层输出正常。
- [ ] A 卷积核对比正常。
- [ ] B 训练 WebSocket 曲线正常。
- [ ] B checkpoint 保存、列表、重新测试正常。
- [ ] B 训练聊天室能多人加入并发送消息。
- [ ] B `@智能助手` 在配置 `ARK_API_KEY` 后可回答。
- [ ] C TF.js 模型加载成功。
- [ ] C Grad-CAM heatmap 和 overlay 正常显示。
- [ ] D 单步反向传播正常。
- [ ] D 决策边界和优化器对比正常。
- [ ] A/B/C/D LLM 未配置 key 时提示清楚。
- [ ] `npm run build` 成功。
- [ ] `mvn test` 成功，或明确说明当前只有编译无测试。
- [ ] Python forward 和 training worker 语法检查通过。
- [ ] `docker compose up --build` 至少在一台机器上验证通过。
- [ ] README、部署说明、用户手册、接口说明、小组分工、AI_USAGE 已更新。
- [ ] 每位成员各自的开发文档已完成。
- [ ] 演示脚本不依赖现场临时下载大文件。



xyp，zhl完成RNN模式和Transformer模式，需要实现帮助手册和大模型按钮。
wlf完成部署和博物馆
lzh完成B模式和最后的文档集成。
每个人27号过后研读自己负责的代码方便展示。
形成开发文档。