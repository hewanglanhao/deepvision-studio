# DeepVision Studio 开发文档

> 本文档用于课程项目答辩与后续维护，不按开发时间顺序记录，而是按项目架构、模块设计、实现难点、部署方案与小组分工组织。

## 1. 文档定位

DeepVision Studio 是一个面向深度学习教学的可视化实验平台，包含网络结构编辑、真实前向计算、训练实验、可视化解释、AI 辅助问答、在线协作与部署。

本文档采用以下写法：

- 以架构与功能模块为主线，说明系统为什么这样设计。
- 每个成员的贡献都区分“人工主导”和“AI 辅助”，AI 不作为独立作者，而是归属到具体成员的研发过程。
- 对工程中可能遇到的关键问题给出合理复盘，包括问题背景、解决策略与代码落点。
- 保留 B/C/D/E 模式、训练协作、解释器等后续章节模板，当前版本重点完成成员 A 王龄锋负责内容。

## 2. 项目概览

### 2.1 项目名称

DeepVision Studio：深度学习算法可视化仿真平台。

### 2.2 核心能力

| 能力 | 说明 | 主要代码位置 |
| --- | --- | --- |
| 首页与模式入口 | 统一进入 A/B/C/D/E、教学文档与 AI 博物馆 | `frontend/src/app/shell/home` |
| A 模式：前向传播实验室 | 编辑 CNN 风格网络、选择样本、执行真实前向传播、观察每层输出 | `frontend/src/app/modes/mode-a` |
| AI 博物馆 | 第一人称漫游 AI 发展史长廊，支持多人联机同馆参观 | `frontend/src/app/modes/ai-museum`，`backend/spring/src/main/java/com/deepvision/studio/museum` |
| 3D 网络显示 | 将网络层、shape、特征图快照映射为 Three.js 3D 场景 | `frontend/src/app/shared/network-3d` |
| 登录注册 | 用户注册、登录、JWT 会话恢复 | `frontend/src/app/core/auth`，`backend/spring/src/main/java/com/deepvision/studio/auth` |
| A 模式历史记录 | 保存网络快照、预览图、参数统计，并支持回溯 | `frontend/src/app/shared/forward`，`backend/spring/src/main/java/com/deepvision/studio/forward` |
| LLM 浮标 | 页面上下文问答、图像上下文、流式输出 | `frontend/src/app/shared/llm`，`backend/spring/src/main/java/com/deepvision/studio/llm` |
| 教学帮助浮标 | 术语高亮、术语检索、教学文档跳转 | `frontend/src/app/shared/teaching`，`frontend/src/app/shell/teaching` |
| 后端网关 | Spring Boot 统一承接认证、持久化、LLM、forward 代理 | `backend/spring` |
| Swagger/OpenAPI 文档 | Spring REST 接口分组、请求/响应模型和 JWT 说明 | `backend/spring/src/main/java/com/deepvision/studio/common/OpenApiConfig.java` |
| Python forward 服务 | 使用 NumPy 执行真实前向传播计算 | `backend/python-forward` |
| Docker 部署 | 前端 Nginx、Spring 后端、Python forward 三容器编排 | `docker-compose.yml`，`frontend/Dockerfile`，`backend/spring/Dockerfile`，`backend/python-forward/Dockerfile` |

## 3. 总体架构设计

### 3.1 架构分层

项目采用“前端可视化 + Spring 业务后端 + Python 计算服务”的分层方式：

```text
Browser
  |
  | Angular SPA
  | - 页面路由
  | - 网络编辑
  | - 2D/3D 可视化
  | - LLM/帮助浮标
  v
Spring Boot Backend
  | - Auth / JWT / H2
  | - A 模式历史记录
  | - LLM 代理与 SSE
  | - Python forward 代理
  v
Python Flask Forward Service
  | - 图执行顺序
  | - shape 校验
  | - Conv/Pool/Dense/Activation/Dropout/Output
  | - 张量统计与可视化数据
```

这样拆分的原因：

- Angular 适合承载复杂交互、局部状态和可视化组件。
- Spring Boot 负责认证、安全、接口、数据库、异常处理、部署配置。
- Python/NumPy 更适合实现张量计算，避免在 Java 或 TypeScript 中重复造低效的数值计算逻辑。
- Spring 作为唯一业务入口代理 Python 服务，前端不直接访问 Python，部署时容器网络和安全边界更清晰。

### 3.2 前端架构

前端按“通用基础设施、共享能力、模式页面、外壳页面”分层：

| 目录 | 职责 |
| --- | --- |
| `core` | 全局 API、认证、登录注册页 |
| `shared` | 多模式复用能力，如网络图、3D 展示、LLM 浮标、教学浮标、forward 客户端 |
| `modes` | A/B/C/D/E 各模式页面与其私有逻辑 |
| `shell` | 首页、教学文档等非模式页面 |
| `public` | 大体积静态资源、样本图片、第三方解释器资源 |

路由入口在 `frontend/src/app/app.routes.ts`，主要路由包括：

| 路径 | 页面 |
| --- | --- |
| `/` | 首页 |
| `/login`、`/register` | 登录注册 |
| `/mode-a` | A 模式前向传播实验室 |
| `/mode-b` | 训练模式 |
| `/mode-c` | CNN 解释器 |
| `/mode-d` | 反向传播/解释模块 |
| `/mode-e` | Transformer 解释器 |
| `/ai-museum` | AI 博物馆 |
| `/network-3d` | 3D 网络显示窗口 |
| `/teaching` | 教学文档 |

### 3.3 后端架构

Spring Boot 后端按业务域划分包：

| 包 | 职责 |
| --- | --- |
| `auth` | 用户、登录注册、JWT、UserDetails |
| `common` | 安全配置、CORS、静态资源映射、健康检查、统一异常 |
| `forward` | A 模式 forward 代理、历史记录、预览图保存 |
| `training` | 训练数据集、任务、检查点、WebSocket 协作 |
| `llm` | 大模型聊天代理与 SSE 流式输出 |
| `museum` | AI 博物馆在线状态 WebSocket |
| `common/OpenApiConfig` | Spring REST API 的 OpenAPI 元信息和 JWT Bearer 鉴权说明 |

数据库当前使用 H2 文件数据库，配置在 `backend/spring/src/main/resources/application.yml`。实体设计基于 JPA，后续可替换为 MySQL/PostgreSQL。

## 4. 小组分工与 AI 使用标注

### 4.1 标注规则

| 标记 | 含义 |
| --- | --- |
| 人工主导 | 需求拆解、架构取舍、关键逻辑、调试与集成由成员完成 |
| AI 辅助 | 使用 AI 辅助生成样式草稿、接口样板、文案、局部算法参考或排错建议 |
| 人工复核 | 对 AI 产出进行修改、测试、合并、删减和工程化落地 |

### 4.2 成员总览

| 成员 | 代号 | 主要职责 | 人工/AI 比例说明 |
| --- | --- | --- | --- |
| 王龄锋 | 成员 A | 项目初始化、整体网站设计、A 模式、登录注册、H2 数据库、LLM 浮标、帮助浮标、Spring/forward 服务、Docker 部署 | 人工主导架构与核心逻辑，AI 辅助 UI 细化、接口样板和文档整理 |
| 李子涵 | 成员 B | 训练实验、训练数据集、训练任务、实验对比、训练运行时 | 待补充 |
| 肖羽平 | 成员 C | CNN/可解释性相关模式、资源迁移与交互解释 | 待补充 |
| 赵红林 | 成员 D | Transformer/反向传播/协作或展示相关模块 | 待补充 |

## 5. 成员 A：王龄锋开发内容

### 5.1 工作范围

成员 A 主要负责项目基础框架和 A 模式相关模块：

| 模块 | 具体内容 | 贡献方式 |
| --- | --- | --- |
| 项目初始化 | 建立前后端目录结构，确定 Angular + Spring Boot + Python 服务拆分 | 人工主导 |
| 模式设计 | 规划 A/B/C/D/E 多模式平台结构，A 模式优先完整实现 | 人工主导，AI 辅助整理交互草图 |
| A 模式页面 | 网络模板、层编辑、样本选择、前向计算、层检查器、卷积核对比、历史记录 | 人工主导核心交互与数据流，AI 辅助局部样式和提示文案 |
| 3D 网络层显示 | Three.js 场景、层几何体、连接线、传播粒子、交互选中、层详情面板 | 人工主导方案和集成，AI 辅助部分 Three.js API 写法 |
| AI 博物馆 | 第一人称 AI 发展史展厅、展品路线、展墙内容、多人联机在线状态 | 人工主导产品设计和 WebSocket 方案，AI 辅助展品文案与 Three.js 局部实现 |
| Spring 后端 | 认证、JWT、安全配置、forward 代理、历史记录、上传图保存、LLM 代理 | 人工主导接口设计与落地 |
| Swagger 接口文档 | Spring REST Controller 注解、DTO Schema、Swagger UI 放行配置 | 人工主导接口分组和描述，AI 辅助注解样板 |
| H2 数据库 | 用户表、A 模式历史记录表、JPA 实体与索引 | 人工主导 |
| 登录注册 | 前端登录注册页、会话恢复、后端密码加密和 JWT | 人工主导，AI 辅助表单样式 |
| LLM 浮标 | 浮动聊天窗口、页面上下文、图像上下文、流式响应 | 人工主导功能设计，AI 辅助 Markdown 渲染与 prompt 文案 |
| 帮助文档浮标 | 术语高亮、教学文档入口、浮标交互 | 人工主导 |
| Docker 部署 | 三服务容器、环境变量、数据卷、Nginx 代理 | 人工主导，AI 辅助命令说明 |

### 5.2 A 模式设计思路

A 模式定位为“前向传播实验室”，将输入图像、每层输出 shape、每层张量可视化、Top-K 结果和公式说明放在同一个操作界面中。

前端主页面位于 `frontend/src/app/modes/mode-a/mode-a-page.component.ts` 与 `.html`。页面被拆成三栏：

| 区域 | 功能 |
| --- | --- |
| 左侧面板 | 输入样本、图像预处理、网络模板选择 |
| 中间画布 | 网络结构编辑、层拖拽/新增/删除、参数编辑、3D 显示入口 |
| 右侧检查器 | 当前层 shape、公式、特征图、通道预览、统计值、最终输出 |

核心数据流如下：

```text
用户选择样本/调整网络
  -> 前端生成 layers + connections + inputTensor
  -> POST /api/forward
  -> Spring 代理到 Python /api/forward
  -> Python NumPy 执行图计算
  -> 返回 layerResults / finalTensor / stats / validationIssues
  -> 前端渲染每层输出、公式、图像、Top-K、shape path
```

### 5.3 A 模式网络结构与前向计算

A 模式支持的层类型包括：

| 层类型 | 作用 |
| --- | --- |
| `input` | 输入图像张量，记录预处理方式 |
| `conv2d` | 卷积，支持 kernel、stride、padding、dilation、activation、多输入/输出通道 |
| `pool2d` | 最大池化或平均池化 |
| `flatten` | 将图像张量转为向量 |
| `dense` | 全连接层 |
| `activation` | 独立激活层 |
| `dropout` | 推理/训练状态下的 dropout 演示 |
| `output` | 输出层与 Top-K 类别展示 |

真实前向计算放在 `backend/python-forward/forward_engine.py`，主要流程为：

1. 构建执行图：根据 `layers` 和 `connections` 建立入边、出边。
2. 拓扑排序：检查循环依赖或非法连接。
3. 参数校验：对 kernel、stride、padding、units、dropout rate 等字段进行约束。
4. 逐层执行：调用对应 operator，如 `run_conv2d_operator`、`run_pool2d_operator`、`run_dense_operator`。
5. 生成可视化数据：输出 shape、张量统计、通道预览、Top-K。

其中卷积计算使用 NumPy 的 `sliding_window_view` 和 `tensordot`，减少 Python 层循环。Dense 层在没有手动权重时使用基于层 ID 的确定性权重生成，保证同一结构重复运行结果稳定。

### 5.4 Spring forward 代理设计

前端只调用 Spring，不直接调用 Python 服务。对应接口在 `backend/spring/src/main/java/com/deepvision/studio/forward/ForwardProxyController.java`：

| 方法 | 路径 | 说明 | 鉴权 |
| --- | --- | --- | --- |
| `GET` | `/api/forward/health` | 检查 Python forward 服务状态 | 公开 |
| `POST` | `/api/forward` | 转发前向传播请求到 Python `/api/forward` | 公开 |

代理层做了三件事：

- 将 Python 服务地址抽成配置项 `deepvision.forward.base-url`，本地和 Docker 环境使用不同地址。
- 设置连接超时与读取超时，避免前端请求无限挂起。
- 捕获 Python 服务不可用和 HTTP 异常，返回更适合前端显示的错误信息。

Python 服务自身接口在 `backend/python-forward/app.py`：

| 方法 | 路径 | 输入 | 输出 |
| --- | --- | --- | --- |
| `GET` | `/api/health` | 无 | `{ ok, service }` |
| `POST` | `/api/forward` | `layers`、`connections`、`inputTensor` | `ForwardPassResult` 风格的 JSON |

### 5.5 A 模式历史记录与 H2 数据库

为了让 A 模式不只是一次性演示，成员 A 设计了历史记录功能。用户登录后可以保存当前网络结构、输入样本、前向传播结果和预览图，并在以后回溯。

相关后端代码：

- `ForwardRecordController.java`
- `ForwardRecord.java`
- `ForwardRecordRepository.java`
- `LocalImageStorage.java`
- `ForwardRecordDtos.java`

数据库实体设计：

| 表/实体 | 字段 | 说明 |
| --- | --- | --- |
| `app_users` / `AppUser` | `id`、`username`、`passwordHash`、`displayName`、`createdAt` | 用户账户 |
| `forward_records` / `ForwardRecord` | `id`、`user_id`、`name`、`templateId`、`datasetName`、`layerCount`、`parameterCount`、`imagePath`、`snapshotJson`、`createdAt` | A 模式保存记录 |

`forward_records` 通过 `user_id` 关联用户，并建立 `user_id, created_at` 索引，用于按用户倒序读取历史记录。

接口设计：

| 方法 | 路径 | 说明 | 鉴权 |
| --- | --- | --- | --- |
| `GET` | `/api/a/forward-records` | 获取当前用户 A 模式历史记录列表 | JWT |
| `POST` | `/api/a/forward-records` | 保存当前 A 模式快照 | JWT |
| `GET` | `/api/a/forward-records/{id}` | 读取某条记录详情并回溯 | JWT |
| `DELETE` | `/api/a/forward-records/{id}` | 删除某条记录 | JWT |

保存时前端会提交 `snapshot` JSON 和 `previewImageDataUrl`。后端将图片从 Data URL 解码为本地文件，路径通过 `/uploads/**` 静态映射访问；完整网络快照则保存为 `snapshotJson`，保持模式结构的灵活性。

### 5.6 登录注册与安全设计

登录注册由 `auth` 包实现：

| 接口 | 说明 |
| --- | --- |
| `POST /api/auth/register` | 注册用户，密码使用 BCrypt 加密 |
| `POST /api/auth/login` | 登录并签发 JWT |
| `GET /api/auth/me` | 读取当前用户信息 |

安全配置位于 `SecurityConfig.java`：

- 使用 `BCryptPasswordEncoder` 存储密码哈希。
- 使用 JWT 进行无状态认证，避免服务端 session。
- 允许前端本地开发端口 `4200/4201/4202` 跨域。
- H2 Console、健康检查、forward、LLM 等接口按开发环境配置放开；A 模式历史记录仍需要用户身份。

前端认证逻辑位于 `frontend/src/app/core/auth`。A 模式页面也内置了保存记录时的登录弹窗，避免用户为了保存实验记录必须先离开当前页面。

### 5.7 Swagger/OpenAPI 接口文档

Spring 后端引入 `springdoc-openapi-starter-webmvc-ui`，只对 Spring REST 接口生成 Swagger/OpenAPI 文档，Python forward 服务不单独生成 Swagger。

相关代码：

- `backend/spring/pom.xml`：加入 `springdoc-openapi-starter-webmvc-ui`。
- `common/OpenApiConfig.java`：配置 API 标题、版本、服务器地址和 JWT Bearer 鉴权方案。
- `common/SecurityConfig.java`：放行 `/v3/api-docs/**`、`/swagger-ui/**`、`/swagger-ui.html`。
- 各 REST Controller：使用 `@Tag`、`@Operation`、`@ApiResponse` 描述接口分组、用途和返回状态。
- 主要 DTO：使用 `@Schema` 描述请求体和响应模型。

Swagger 分组覆盖：

| 分组 | 代码位置 | 内容 |
| --- | --- | --- |
| `Health` | `HealthController` | Spring 健康检查 |
| `Auth` | `AuthController` | 注册、登录、当前用户 |
| `Mode A Forward` | `ForwardProxyController` | Python forward 健康检查和前向计算代理 |
| `Mode A Records` | `ForwardRecordController` | A 模式历史记录的列表、保存、详情、删除 |
| `LLM` | `LlmController` | 普通聊天和 SSE 流式聊天 |
| `Training` | `TrainingController` | 数据集、训练任务、checkpoint、实验控制和协作房间查询 |

本地访问地址：

```text
http://127.0.0.1:8080/swagger-ui/index.html
http://127.0.0.1:8080/v3/api-docs
```

Docker 或 Nginx 代理后，可通过前端同源地址访问：

```text
http://localhost:4200/swagger-ui/index.html
http://localhost:4200/v3/api-docs
```

其中需要用户身份的接口在 OpenAPI 中使用 `bearerAuth` 标记，调用时在 Swagger UI 的 Authorize 中填入登录接口返回的 JWT。

### 5.8 3D 网络层显示

3D 网络显示位于 `frontend/src/app/shared/network-3d`，入口路由为 `/network-3d`。A 模式点击“3D化显示”时，会把当前网络快照写入 `sessionStorage`，再打开独立窗口展示。

设计重点：

- 使用 Three.js 渲染网络层，避免把 3D 结构塞进普通 DOM 导致性能和透视关系难以控制。
- 根据层类型选择不同几何表达：卷积/池化为特征图堆叠，Flatten 为条带，Dense/Output 为单元网格。
- 把前向传播结果中的特征图预览贴到 3D 平面上，使 3D 场景不只是结构图，而能显示真实计算结果。
- 增加 OrbitControls、单击选中、双击聚焦、传播粒子和右侧详情面板。

工程难点与解决：

| 难点 | 解决方式 |
| --- | --- |
| 不同层 shape 差异很大，直接按真实尺寸渲染会过大或过小 | 在 `network-3d-layout.ts` 中将 shape 映射为受限宽高深，保留比例但限制视野 |
| 特征图通道多，全部渲染会卡顿 | 最多展示部分通道，并用 `+N ch` 标记剩余通道 |
| 3D 场景与 A 模式页面状态解耦 | 使用 `NETWORK_3D_SESSION_KEY` 传递快照，独立窗口只读快照 |
| Three.js 资源泄漏 | 组件销毁时 dispose renderer、controls、geometry、material |

### 5.9 AI 博物馆与联机参观设计

AI 博物馆入口为 `/ai-museum`，代码位于 `frontend/src/app/modes/ai-museum`。前端使用 Three.js 构建第一人称展厅：用户进入后可以用 WASD/方向键移动、鼠标控制视角、Shift 加速，靠近展墙时右侧导览面板显示当前展品说明。

博物馆展品按 AI 发展脉络组织，包括人工神经元、图灵测试、Dartmouth、感知机、ELIZA、专家系统、AI 寒冬、Hopfield 网络、反向传播、Q-learning、SVM、LeNet、RNN/LSTM、AlexNet、YOLO、深度强化学习、Transformer、LLM、RLHF、GRPO、多模态与智能体工作流等。每个展品包含年份、标题、副标题、说明、要点、标签、来源、颜色主题和 3D 位置。

前端设计重点：

| 能力 | 实现方式 |
| --- | --- |
| 第一人称漫游 | Three.js `PointerLockControls`，锁定鼠标后移动相机 |
| 展厅结构 | 使用长廊、墙面、地面时间线、年份标记和展墙卡片组织空间 |
| 展品内容 | 将展品定义为结构化 `MuseumExhibit` 数组，便于维护和扩展 |
| 动态展品 | 根据展品类型生成不同 3D artifact，例如注意力环、token 方块、强化学习交互符号等 |
| 导览面板 | 每帧根据相机位置寻找最近展品，距离足够近时显示说明 |
| 性能控制 | 限制展厅宽长、相机边界、像素比和动画对象数量，避免 3D 场景过重 |

联机能力由 Spring WebSocket 实现，后端代码位于 `MuseumPresenceHandler.java`，注册路径为 `/api/museum/presence`。用户进入博物馆后，前端建立 WebSocket 连接，并定期发送当前位置与朝向：

```text
Browser A/B/C
  -> ws://host/api/museum/presence?token=...
  -> Spring MuseumPresenceHandler
  -> room 分配、join/pose/leave 广播
  -> 其他浏览器渲染远程参观者 avatar
```

联机协议：

| 消息类型 | 方向 | 说明 |
| --- | --- | --- |
| `welcome` | 服务端 -> 当前用户 | 返回自己的 `selfId`、房间号、房间人数上限和当前参与者 |
| `join` | 服务端 -> 房间其他用户 | 新用户进入房间 |
| `pose` | 双向 | 客户端发送位置，服务端广播给其他用户 |
| `leave` | 服务端 -> 房间其他用户 | 用户断开连接，移除 avatar |

后端联机设计：

- 每个房间最多 8 人，控制同一展厅的远程 avatar 数量。
- 登录用户通过 JWT 识别显示名，未登录用户以“游客 N”身份进入。
- 服务端保存 `Participant` 的房间、颜色、坐标、朝向和更新时间。
- 坐标在后端做 clamp，防止异常客户端发送离谱位置。
- 房间无人时自动移除，避免长期占用内存。

### 5.10 LLM 浮标设计

LLM 浮标位于 `frontend/src/app/shared/llm`，后端位于 `backend/spring/src/main/java/com/deepvision/studio/llm`。

前端能力：

- 固定在页面右下角，可展开/收起。
- 支持普通问答和“传入页面上下文”两种模式。
- A 模式提供专用 prompt，如解释当前层、卷积核差异、输出 shape、答辩总结。
- 支持传入文本上下文和最多 4 张图像 URL。
- 前端做轻量 Markdown 渲染，展示列表、代码和加粗文本。

后端接口：

| 方法 | 路径 | 说明 |
| --- | --- | --- |
| `POST /api/llm/chat` | 普通 JSON 聊天响应 |
| `POST /api/llm/chat/stream` | SSE 流式聊天响应 |

后端通过 `LlmChatClient` 对接火山方舟兼容接口，配置项包括：

- `ARK_BASE_URL`
- `ARK_API_KEY`
- `ARK_MODEL`
- `ARK_CONNECT_TIMEOUT_SECONDS`
- `ARK_READ_TIMEOUT_SECONDS`

### 5.11 帮助文档浮标设计

帮助文档浮标位于 `frontend/src/app/shared/teaching`。其作用是降低深度学习术语门槛：

- 页面中重要术语通过 `TeachingTermDirective` 标注。
- 浮标开启后，用户可看到术语提示，并跳转 `/teaching` 教学文档。
- 浮标不改变主页面路由和实验状态。

### 5.12 页面美化与交互一致性

成员 A 负责整体网站视觉风格和 A 模式界面美化，主要原则：

- 顶栏统一显示当前模式、用户状态和关键操作。
- A 模式采用工作台式布局。
- 面板密度较高，但用标题、标签、状态条和弹窗区分信息层级。
- 历史记录、样本选择、卷积核对比、图片查看器均使用浮层/抽屉，保持主画布不频繁跳页。
- AI 博物馆采用沉浸式全屏布局，与工作台页面区分开，使首页既有实验入口也有展示型入口。
- LLM 浮标和帮助浮标固定在右下角，但错开位置，避免互相遮挡。

AI 辅助主要用于局部 CSS 草稿、按钮文案备选和空状态提示；最终布局、功能取舍和与实际数据绑定由人工完成。

### 5.13 Docker 部署方案

部署方案使用 `docker-compose.yml` 编排三个服务：

| 服务 | 说明 | 端口 |
| --- | --- | --- |
| `frontend` | Angular production build，由 Nginx 托管 | 宿主机 `4200` -> 容器 `80` |
| `spring-backend` | Spring Boot 后端，连接 H2 文件库，代理 Python 和 LLM | 容器内部 |
| `python-forward` | Flask + NumPy 前向计算服务 | 容器内部 `5000` |

关键环境变量：

| 变量 | 作用 |
| --- | --- |
| `DEEPVISION_JWT_SECRET` | JWT 密钥 |
| `DEEPVISION_FORWARD_BASE_URL` | Spring 访问 Python forward 服务地址 |
| `DEEPVISION_DB_URL` | H2 数据库地址 |
| `DEEPVISION_UPLOAD_ROOT` | 上传文件目录 |
| `ARK_API_KEY` | LLM 服务 API Key |
| `ARK_MODEL` | 默认大模型 |

数据卷：

- `spring-data`：H2 数据库。
- `spring-uploads`：A 模式历史记录预览图。
- `spring-datasets`：训练/数据集资源。
- `spring-training-jobs`：训练任务输出。

本地启动：

```powershell
docker compose up --build
```

访问：

```text
http://localhost:4200
http://localhost:4200/api/health
```

前端 Nginx 同时代理 `/swagger-ui/**` 和 `/v3/api-docs/**` 到 Spring，容器环境下 Swagger UI 可通过 `http://localhost:4200/swagger-ui/index.html` 访问。

### 5.14 成员 A 攻克的主要难点

| 难点 | 背景 | 解决方案 | 体现的工程能力 |
| --- | --- | --- | --- |
| A 模式既要可视化又要真实计算 | 如果计算和渲染都放在前端，计算边界和结果来源不清晰 | Python NumPy 执行图计算，Spring 只做代理，前端渲染真实结果 | 架构拆分、跨语言服务集成 |
| 图像张量过大导致页面卡顿 | 原图或中间特征图可能很大 | 前端设置预览尺寸上限，Python 返回可视化摘要，真实 tensor 与预览分离 | 性能优化、渲染控制 |
| 早期特征图渲染方式卡顿 | 初版为了展示卷积效果，曾把卷积输出拆成大量小块 DOM 节点逐个渲染，再在页面上拼成图像；当图片尺寸稍大时会生成数百甚至上千个 HTML 元素，浏览器布局和重绘压力很大 | 改为先拿到完整计算结果，再用 Canvas/ImageData 一次性合成为图片或 Data URL，页面只渲染一个 `img` 或少量通道预览节点 | 前端性能分析、渲染模型优化 |
| 网络层 shape 易出错 | stride/padding/kernel/dilation 组合容易得到非法输出 | Python 后端统一校验并返回 `validationIssues`，前端高亮错误字段 | 数据校验、错误反馈 |
| 3D 网络展示信息不足 | 只画层结构无法说明网络运行过程 | 将 layerSnapshots、shape、特征图、Top-K 绑定进 3D 场景 | 可视化抽象能力 |
| AI 博物馆需要兼顾展示和实时联机 | 静态时间线缺乏沉浸感，但 3D + WebSocket 容易让状态管理复杂 | 前端用结构化展品数组驱动 Three.js 场景，后端用房间状态管理 `join/pose/leave`，客户端只渲染远程 avatar | 3D 交互设计、实时通信 |
| 历史记录要保存复杂状态 | 网络结构、输入、输出、预览图不是固定表结构 | 元数据结构化入表，完整快照以 JSON 保存，图片单独落盘 | 数据建模、持久化设计 |
| LLM 不能直接暴露 Key 到前端 | API Key 放前端有泄漏风险 | Spring 后端代理 LLM，并支持 SSE | 安全意识、后端接口设计 |
| Docker 中服务地址变化 | 本地 Python 是 `127.0.0.1:5000`，容器内需走服务名 | 通过 `DEEPVISION_FORWARD_BASE_URL` 区分环境 | 部署配置能力 |

### 5.15 成员 A 的 AI 使用复盘

#### 5.15.1 总体边界

| 类型 | 人工负责 | AI 辅助 | 最终责任 |
| --- | --- | --- | --- |
| 需求与模式设计 | 确定 A/B/C/D/E 平台结构，决定 A 模式做成“可编辑网络 + 真实 forward + 3D 展示” | 辅助整理功能清单、生成可选页面文案 | 人工筛选功能，删除超出课程规模的设计 |
| 架构拆分 | 决定 Angular、Spring、Python forward 三服务拆分，确定 Spring 作为统一 API 入口 | 辅助比较前后端通信方式和 Docker 说明表达 | 人工根据本地运行、部署和代码维护成本做取舍 |
| 数据结构 | 定义 `layers`、`connections`、`inputTensor`、`ForwardPassResult`、历史记录 snapshot 的核心字段 | 辅助补全 TypeScript/Java DTO 样板 | 人工调试前后端字段一致性，修正 shape、tensor、preview 数据 |
| 业务代码 | 编写和整合 A 模式状态流、保存回溯、登录注册联动、LLM 上下文、3D 快照传递 | 辅助生成局部函数草稿、表单代码、错误处理模板 | 人工逐段合并、运行、调试，并按项目结构重命名和拆分 |
| 算法实现 | 明确要支持 Conv/Pool/Dense/Activation/Dropout/Output，定义每层输入输出 | 辅助提供 NumPy 向量化卷积、softmax、统计值等参考写法 | 人工校验输出 shape、调通与前端可视化的数据契约 |
| UI 与样式 | 确定工作台三栏布局、右下角浮标、历史记录抽屉和 3D 窗口入口 | 辅助生成 CSS 草稿、空状态和按钮文案 | 人工根据真实页面密度、中文长度和交互状态反复调整 |
| 文档整理 | 确定文档结构、接口表、工程难点、分工表达 | 辅助语言组织和表格整理 | 人工核对代码路径、删除夸大表述 |

#### 5.15.2 重点业务代码分工

| 模块 | 人工设计与决策 | AI 参与方式 | 人工修改/落地点 |
| --- | --- | --- | --- |
| `ModeAPageComponent` | 决定 A 模式页面状态：网络层、连接、样本、当前输入、forward 结果、记录抽屉、弹窗状态统一放在页面组件中管理 | AI 辅助生成部分 getter、表单绑定、弹窗结构和错误提示草稿 | 人工把草稿接入真实 `SimEngine`、`ForwardBackendService`、`ForwardRecordService`，并处理自动计算、防抖、取消和回溯逻辑 |
| A 模式 forward 请求 | 决定前端只提交抽象网络结构和输入张量，不把计算逻辑放在 DOM 渲染层 | AI 辅助生成请求体类型和 service 方法 | 人工确定 `/api/forward` 契约，并让 Spring 代理到 Python，避免前端直接访问 Python 服务 |
| Python forward 引擎 | 决定后端必须返回每层 `layerResults`、`shapePath`、`validationIssues`、`stats`，便于前端解释 | AI 辅助提供 NumPy `sliding_window_view`、`tensordot`、softmax、Top-K 等实现参考 | 人工修正层类型、参数字段、错误信息和前端需要的可视化结构 |
| 特征图渲染优化 | 人工发现初版大量 DOM 小块拼图导致严重卡顿，并决定改成 Canvas/ImageData 一次性生成图片 | AI 辅助确认浏览器 DOM 数量、重排重绘和 Canvas 渲染的性能差异 | 人工改造 `tensorToImageDataUrl`、`grayValuesToImageDataUrl`、通道预览缓存等前端逻辑 |
| 3D 网络显示 | 决定通过快照传递 A 模式结果，3D 页面只负责展示，不重新计算 | AI 辅助 Three.js 几何体、材质、OrbitControls 写法 | 人工设计层类型到几何体的映射、shape 缩放策略、选中/聚焦/传播粒子交互 |
| AI 博物馆 | 人工决定做成第一人称时间长廊，设计展品顺序、空间布局、移动方式和联机参观目标 | AI 辅助生成部分展品说明、Three.js 展品 artifact 草稿和 WebSocket 消息样板 | 人工整合 `AiMuseumPageComponent`、`MuseumPresenceHandler`，调试 pointer lock、位置广播、房间人数和 avatar 显示 |
| 登录注册与历史记录 | 决定保存记录必须绑定用户，历史记录只返回当前用户数据 | AI 辅助生成 Controller/DTO/Repository 样板 | 人工补充 BCrypt、JWT、Principal 取用户、Data URL 图片落盘和 `/uploads/**` 映射 |
| LLM 浮标 | 决定 LLM 不是单独页面，而是作为全局浮标接入 A 模式上下文 | AI 辅助 prompt、Markdown 渲染和 SSE 接收草稿 | 人工控制上下文摘要、图片数量、流式异常处理和后端 API Key 代理 |
| Docker 部署 | 决定拆为前端、Spring、Python forward 三容器，并使用卷保存 H2/上传文件 | AI 辅助命令说明和环境变量说明 | 人工调通容器内服务名、端口映射和数据卷 |

#### 5.15.3 典型 AI 辅助但人工主导的迭代案例

以 A 模式卷积结果渲染为例，初版思路是把每个卷积输出单元都当成页面元素渲染，再通过大量小块拼接出特征图。这个方案在小样本下能直观看到“每个格子”的值，但图片稍大时会创建大量 DOM 节点，导致页面滚动、刷新和参数调整都变慢。

后续重构时，渲染方案改为保留张量数组，展示阶段一次性写入 Canvas，再转换成图片地址交给 `<img>` 渲染；多通道特征图默认只展示前几个通道，完整通道通过弹窗查看。

## 6. 成员 B：李子涵开发内容（预留）

> 后续补充训练实验、训练数据集、训练任务、实验对比、训练协作等内容。

建议补充结构：

- 负责模块概述
- 训练任务架构
- 数据集上传与内置数据集
- WebSocket/SSE 训练进度
- checkpoint 与实验对比
- 工程难点
- 人工与 AI 使用比例说明

## 7. 成员 C：肖羽平开发内容（预留）

> 后续补充 CNN 解释器、可视化解释、资源迁移与交互说明。

建议补充结构：

- 负责模块概述
- CNN Explainer 集成或重构策略
- 静态资源管理
- 与 Angular 页面融合
- 工程难点
- 人工与 AI 使用比例说明

## 8. 成员 D：赵红林开发内容

### 8.1 负责模块概述

成员 D 负责以下模块的设计与实现：

| 模块 | 路径 | 说明 |
| --- | --- | --- |
| 模式 E：反向传播可视化 | `frontend/src/app/modes/mode-e/` | MLP 前向/反向传播引擎、逐层子步骤动画、神经元权重连线图、决策边界、优化器对比 |
| 模式 F：RNN 循环神经网络 | `frontend/src/app/modes/mode-f/` | Tanh RNN + BPTT 引擎、时间展开可视化、序列分类数据集 |
| 教学文档（训练与优化 + 序列模型） | `frontend/src/app/shared/teaching/teaching-glossary.ts` | SGD/Momentum/Adam/反向传播/梯度下降/学习率/ReLU/Sigmoid/Tanh + RNN Cell/BPTT/隐状态 等 12 个术语 |
| AI 助手接入（E/F） | 模式 E/F 的 Shell 组件 | 模式专属系统提示词、上下文数据收集、快捷提问 |

### 8.2 模式 E：反向传播可视化

#### 8.2.1 核心设计

模式 E 的核心目标是**展示单步训练中数据如何从前向后流动、梯度如何从后向前回传、参数如何被更新**。

采用纯 TypeScript 在浏览器内实现完整的前向传播、损失计算、反向传播和参数更新引擎（`mode-e-backprop-engine.ts`，约 650 行），零外部依赖。

**支持的层类型**：Dense（全连接）、Activation（ReLU/Sigmoid/Tanh）、Output（Softmax）

**反向传播实现**：
- Dense 层：dW = a_prev^T · dZ, db = Σ dZ, dA_prev = dZ · W^T
- ReLU：dZ = dA · (Z > 0)
- Sigmoid：dZ = dA · σ(Z) · (1 - σ(Z))
- Tanh：dZ = dA · (1 - tanh²(Z))
- Softmax + CrossEntropy 合并：dZ = softmax(Z) - y_onehot

**优化器**：SGD（基础梯度下降）、Momentum（动量累积）、Adam（自适应学习率 + 动量）

**数据集生成器**：XOR 四团分布、同心圆、高斯团三类，均支持噪声参数控制

#### 8.2.2 可视化设计

模式 E 采用三大可视化面板：

**神经元权重连线图**（Overview 组件）：SVG 绘制每层神经元节点和层间全连接权重边。通过子步骤动画逐层揭示前向/反向传播过程——蓝线=前向流动、橙线=反向梯度回传、绿线=参数更新。鼠标悬停连线弹出权重值浮层，选中神经元高亮其入边/出边。

**浮层图表面板**（FloatingCharts 组件）：左侧悬浮面板包含数据集散点图（叠加决策边界半透明色块）、损失曲线对比图（支持多条历史曲线叠加、单条删除）、当前预测状态。点击图表可弹出居中放大弹窗。

**控制面板**（ControlPanel 组件）：预设网络选择、激活函数切换（ReLU/Sigmoid/Tanh）、优化器切换（SGD/Momentum/Adam）、学习率调节、训练步数设置、播放速度控制。

#### 8.2.3 子步骤动画状态机

为实现"单步训练"中逐层流动的教学效果，设计了 `SubStep` 状态机：

```text
idle → forward-0-1 → forward-1-2 → ... → loss → backward-2-1 → backward-1-0 → update-1 → update-2 → done
```

每个子步骤停留至用户点击"继续"按钮，进度条实时显示当前位置。连续播放模式每 N 毫秒自动推进。

#### 8.2.4 决策边界计算

每 25 步训练后用全部样本计算平均损失和整体准确率，同时用 50×50 网格计算模型对二维空间的分类判断，在散点图上叠加半透明色块展示决策边界。数据点坐标根据实际数据集范围动态映射（带 8% 边距），解决了同心圆等数据集点溢出坐标轴的问题。

#### 8.2.5 损失曲线对比

训练完成指定步数后自动保存当前平滑损失曲线（含优化器和激活函数标签）。曲线图支持多条历史曲线叠加显示，颜色自动分配不重复。每条已保存曲线可单独删除（图例旁 x 按钮）。

### 8.3 模式 F：RNN 循环神经网络

#### 8.3.1 核心设计

模式 F 实现一个简单的 Tanh RNN + BPTT（穿越时间的反向传播）引擎（`mode-f-rnn-engine.ts`，约 180 行），支持 SGD/Momentum/Adam 三种优化器。

**前向传播**：对长度为 T 的序列，从初始零隐状态开始，每步计算 h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)，最后时间步通过 softmax 输出分类概率。

**BPTT**：从最后时间步的输出梯度开始，沿时间轴反向传播，同时累积各时间步对 W_xh、W_hh、W_hy 和偏置的梯度。

**数据集**：三个简单序列分类任务——延迟记忆（echo）、XOR 记忆、交替检测，均为 4 步序列，200 个样本。

#### 8.3.2 可视化设计

时间展开图展示 RNN 在各时间步的隐状态向量（彩色条形图）和输出概率。每个 Cell 方框代表一个时间步，箭头串联隐状态传递路径。

### 8.4 共享组件复用

成员 D 开发的模式 E/F 复用了项目中多个共享组件：

| 复用组件 | 使用位置 | 说明 |
| --- | --- | --- |
| PlatformTopbarComponent | E/F Shell | 统一顶栏导航 |
| LlmFloatingAssistantComponent | E/F Shell | AI 浮层助手，配置模式专属提示词和上下文 |
| TeachingSearchFabComponent | E/F Shell | 教学文档浮层按钮 |
| TeachingTermDirective | E 控制面板 | 优化器和激活函数按钮的高亮术语 |

### 8.5 教学文档贡献

在共享的教学词典（`teaching-glossary.ts`）中新增 12 个术语，归属两个分类：

**训练与优化**（模式 E）：SGD 优化器、Momentum 优化器、Adam 优化器、反向传播、梯度下降、学习率、ReLU、Sigmoid、Tanh

**序列模型**（模式 F）：RNN Cell、BPTT、隐状态

每个术语包含标题、别名、分类、摘要和多段详细说明，覆盖原理公式、优缺点分析和在对应模式中的教学应用场景。

### 8.6 工程难点

1. **纯 TypeScript 矩阵运算**：前向/反向传播引擎不依赖任何数值计算库，手动实现了矩阵乘法、转置、Hadamard 积、softmax 等全套运算，需处理维度对齐和数值稳定性。

2. **子步骤动画状态管理**：在单步训练中拆分出逐层子步骤（前向各层对→损失→反向各层对→更新），用 Angular signal 驱动的状态机控制动画节奏和 UI 同步。

3. **决策边界实时渲染**：50×50 网格需要 2500 次前向传播计算，权衡精度与性能后每 25 步更新一次，数据点坐标根据实际范围动态映射避免溢出。

4. **激活函数与决策边界的关系**：ReLU 产生分段直线边界、Sigmoid 产生弧线边界——这一特性在同心圆数据集上表现突出，需要选择合适的预设网络架构（Sigmoid 16 隐藏单元）才能收敛。

### 8.7 AI 使用复盘

- **人工主导**：引擎数学公式推导、子步骤状态机设计、可视化布局决策、同心圆/ReLU 等训练调参
- **AI 辅助**：组件样板代码、CSS 样式微调、字符串批量替换（mode-d→mode-e 迁移）、文案润色
- **AI 工具使用记录**：见 `AI_USAGE.md`

- 所有新模式优先放入 `frontend/src/app/modes/mode-*`，公共能力再提取到 `shared`。
- 后端新增接口应按业务域建包，不建议把 Controller 全部放在同一目录。
- 如果 H2 数据量或并发压力增大，可迁移到 MySQL/PostgreSQL，JPA 实体可基本保留。
- LLM 接口应继续由后端代理，避免在前端暴露 API Key。
- 课程答辩前建议准备一条固定演示路径：登录 -> A 模式选择样本 -> 修改卷积核 -> 执行 forward -> 打开 3D -> 保存历史记录 -> 打开 LLM 浮标解释当前层。
