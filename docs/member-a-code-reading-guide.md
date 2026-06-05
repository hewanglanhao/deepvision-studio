# 成员 A 代码理解与面试讲解指南

本文只围绕成员 A 负责的代码展开，目标不是把每一行代码翻译一遍，而是帮助你建立一条清晰的阅读路线：从页面入口开始，理解数据结构如何流动，用户操作如何触发业务逻辑，前端、Spring 后端和 Python 计算服务如何配合，最后形成可以在面试中讲清楚的实现思路。

## 1. 先建立整体心智模型

成员 A 的核心工作可以概括为一句话：

> 搭建 DeepVision Studio 的基础架构，并完整实现 A 模式“前向传播实验室”以及它依赖的登录、记录保存、后端代理、Python forward、3D 展示、LLM 助手、教学浮标、AI 博物馆和 Docker 部署能力。

不要一上来就从 `mode-a-page.component.ts` 的第一行读到最后一行。这个文件很大，如果没有主线会很容易迷路。更好的读法是先把系统分成三层：

```text
Angular 前端
  - 路由和页面入口
  - A 模式网络编辑、样本选择、结果展示
  - 3D 展示、LLM 浮窗、历史记录 UI

Spring Boot 后端
  - 登录注册和 JWT
  - A 模式记录持久化
  - 转发 forward 请求到 Python
  - 转发 LLM 请求，避免 API Key 暴露在前端

Python forward 服务
  - 根据 layers + connections + inputTensor 构建执行图
  - 按拓扑顺序执行 Conv/Pool/Dense/Activation/Dropout/Output
  - 返回每层 shape、tensor、统计信息、可视化摘要和错误提示
```

面试时可以先讲这层架构，因为它解释了为什么不是把所有计算都写在前端：前端负责交互和可视化，Spring 负责统一 API、安全和持久化，Python/NumPy 负责真实张量计算。

## 2. 第一站：从路由找到 A 模式入口

先看 `frontend/src/app/app.routes.ts`。

这里定义了项目所有页面入口。和成员 A 最相关的路由有：

| 路由 | 入口组件 | 你要理解的作用 |
| --- | --- | --- |
| `/mode-a` | `ModeAPageComponent` | A 模式前向传播实验室主页面 |
| `/network-3d` | `Network3dViewerComponent` | A 模式把当前网络快照传给 3D 页面展示 |
| `/login`、`/register` | `AuthPageComponent` | 登录注册页面 |
| `/ai-museum` | `AiMuseumPageComponent` | AI 博物馆沉浸式展示 |
| `/teaching` | `TeachingDocPageComponent` | 教学文档入口 |

你读 A 模式时，真正的起点是：

```text
app.routes.ts
  -> /mode-a
  -> frontend/src/app/modes/mode-a/mode-a-page.component.ts
```

这个组件是 A 模式的“总调度器”。它不只是页面展示，还管理网络层数组、连接关系、当前输入图片、后端 forward 结果、选中层、历史记录弹窗、LLM 上下文和 3D 快照。

## 3. 第二站：先理解公共数据结构，而不是先读页面细节

读 `frontend/src/app/shared/simulation/sim-models.ts`。

这个文件定义了 A 模式前后端共同理解的核心数据结构。你要重点看四类类型。

第一类是网络结构：

```text
NetworkLayer
Connection
LayerType
InputLayerParams
Conv2DLayerParams
Pool2DLayerParams
DenseLayerParams
OutputLayerParams
```

A 模式里所谓“搭网络”，本质就是维护一个 `NetworkLayer[]` 和一个 `Connection[]`。

`NetworkLayer` 代表一层，例如 input、conv2d、pool2d、flatten、dense、activation、dropout、output。每层都有：

```text
id       唯一标识
type     层类型
name     页面显示名称
inputs   上游层 id
params   当前层参数
```

`Connection` 只记录 `{ from, to }`，用于表达层之间的边。

第二类是输入和张量：

```text
ForwardTensor
ForwardInputAsset
PreparedInputAsset
TensorShape
```

`ForwardTensor` 是前后端传递张量的基本格式：

```text
kind    scalar/vector/matrix/tensor3d
shape   例如 [32, 32, 3]
values  一维数组，保存真实数值
```

`ForwardInputAsset` 则是前端对“当前输入图片”的包装：既保存原始图像张量，也保存预处理后的张量。这样页面既能展示原图，也能把真正计算用的 tensor 发给后端。

第三类是 forward 返回结果：

```text
ForwardPassResult
ForwardLayerResult
LayerVisualization
TensorStats
LayerValidationIssue
```

这组类型决定了 Python 后端要返回什么。每一层不仅返回 tensor，还返回：

```text
inputShapes / outputShape
shapeLabel
paramsSummary
stats
visualization
warnings
```

所以 A 模式右侧检查器、shape 路径、特征图、Top-K、错误提示都来自同一个 `ForwardPassResult`。

第四类是模型模板：

```text
ModelTemplate
LayerDraft
```

模板不是直接写死在 HTML 里，而是以结构化数组形式定义。页面选择模板时，会把 `LayerDraft[]` 转成真正带 id 的 `NetworkLayer[]`。

## 4. 第三站：理解 SimEngine 的角色

读 `frontend/src/app/shared/simulation/sim-engine.ts`。

这个文件名字叫 engine，但在 A 模式里它不是最终真实 forward 计算引擎。它主要做三件事：

1. 提供模型模板。
2. 处理输入图像预处理。
3. 在前端做轻量 shape 推导和参数统计。

### 4.1 模型模板

入口是 `SimEngine.templates()`。

这里定义了几个可选模板，例如：

```text
MLP Basic
CNN Classic
Residual CNN
Analyzer Lite
CSV / Tabular MLP
```

以 `CNN Classic` 为例，结构是：

```text
Input
  -> Conv 1
  -> Pool 1
  -> Conv 2
  -> Flatten
  -> Dense 1
  -> Output
```

页面刚进入 A 模式时，会调用 `applyTemplate()`，把模板层复制出来，再重建线性连接。

### 4.2 输入图片预处理

重点看：

```text
createForwardInputAssetFromImageData()
prepareInputTensor()
imageDataToRgbTensor()
resizeTensorNearest()
normalizeValues()
```

用户选择内置样本或上传图片后，页面会先把图片解码成 `ImageData`，再由 `SimEngine` 转成 `ForwardTensor`。之后根据 input 层的 preprocessing 配置做：

```text
颜色模式转换
resize
invert
normalize
```

这里的关键点是：前端预处理后的 tensor 才是传给 Python forward 的输入。

### 4.3 shape 和参数统计

重点看：

```text
inferLayerOutputShape()
parameterCount()
rebuildLinearConnections()
```

这部分是前端为了即时展示网络规模和 3D 页面兜底 shape 所做的轻量推导。真正带错误校验的 forward 仍然由 Python 服务完成。

面试时可以这样讲：

> SimEngine 在前端负责“准备和预估”，Python forward 负责“真实执行”。这样页面可以快速显示模板、参数量和基础 shape，同时避免把复杂数值计算散落在 DOM 逻辑里。

## 5. 第四站：进入 A 模式主组件的阅读路线

主文件是 `frontend/src/app/modes/mode-a/mode-a-page.component.ts`。

这个文件很大，建议按以下顺序读。

### 5.1 先读状态字段

在类开头附近先找这些字段：

```text
modelTemplates
selectedTemplateId
layers
connections
selectedLayerId
currentInputAsset
forwardResult
forwardLayerShapeMap
forwardBusy
forwardBackendError
forwardRecords
recordBusy
recordError
recordSuccess
```

这些字段能解释页面的主要状态：

| 状态 | 含义 |
| --- | --- |
| `layers` | 当前网络层列表 |
| `connections` | 当前层之间的连接 |
| `selectedLayerId` | 当前右侧检查器正在看的层 |
| `currentInputAsset` | 当前样本或上传图片的输入数据 |
| `forwardResult` | Python 后端返回的完整前向传播结果 |
| `forwardLayerShapeMap` | 每层 id 到 shape 文本的映射 |
| `forwardRecords` | 当前登录用户保存过的 A 模式历史记录 |

只要这几个状态理解了，A 模式的大部分 UI 都能解释清楚。

### 5.2 再读初始化流程

看 `ngOnInit()`。

初始化的大致链路是：

```text
ngOnInit()
  -> 根据路由设置 mode
  -> 恢复登录态 authSvc.restoreSession()
  -> 监听当前用户 user$
  -> applyTemplate()
  -> loadLocalImageManifest()
  -> 用户已登录时 loadForwardRecords()
```

也就是说，页面启动时先建立默认网络，再准备样本，再尝试恢复用户和历史记录。

### 5.3 读模板应用和网络编辑

重点看：

```text
applyTemplate()
addLayer()
deleteSelectedLayer()
moveSelectedLayer()
onLayersReordered()
onNewLayerDropped()
rebuildTopology()
defaultLayer()
```

这里的实现思路是：

1. 用户选择模板时，`applyTemplate()` 把模板层复制为真实 `layers`。
2. 每一层分配数字 id。
3. `rebuildTopology()` 让当前网络保持线性结构。
4. 增删改层后调用 `runForward()`，让结果进入“需要重新计算”或“立即计算”的状态。

A 模式当前主要以线性网络为主，所以 `rebuildTopology()` 会让每层输入指向前一层，并用 `SimEngine.rebuildLinearConnections()` 生成连接。

面试时可以这样讲：

> 页面编辑网络时，实际修改的是 `layers` 数组和 `connections` 数组。UI 只是这些结构的可视化表现。每次结构或参数变化后，都通过统一入口 `runForward()` 触发后续计算，而不是每个按钮各写一套计算逻辑。

### 5.4 读输入图片选择和上传

重点看：

```text
loadLocalImageManifest()
chooseLocalImageSample()
handleImageUpload()
decodeAndResizeImage()
rebuildInputAsset()
```

输入来源有两种：

```text
内置样本：frontend/public/mode-a/samples/manifest.json
用户上传：浏览器 FileReader 读取图片
```

共同流程是：

```text
图片 URL / Data URL
  -> decodeAndResizeImage()
  -> 得到 ImageData + previewUrl
  -> rebuildInputAsset()
  -> SimEngine.createForwardInputAssetFromImageData()
  -> 得到 originalTensor + prepared.tensor
  -> runForward()
```

这里有一个工程细节：图片会先限制最大边长，避免用户上传过大的图导致前端和 Python 传输/计算过重。

### 5.5 读前向传播主流程

这是 A 模式最核心的链路。重点看：

```text
runForward(force = false)
applyForwardResult()
ForwardBackendService.executeForward()
```

前端请求体是：

```text
{
  layers,
  connections,
  inputTensor: currentInputAsset.prepared.tensor
}
```

完整链路是：

```text
用户点击开始计算或参数变化
  -> ModeAPageComponent.runForward()
  -> ForwardBackendService.executeForward()
  -> POST /api/forward
  -> Spring ForwardProxyController
  -> Python /api/forward
  -> execute_forward_graph()
  -> 返回 ForwardPassResult
  -> applyForwardResult()
  -> 页面刷新 shape、特征图、Top-K、错误提示
```

`runForward()` 里有几个值得面试讲的点：

| 设计 | 作用 |
| --- | --- |
| `force` 参数 | 区分“参数已更新，等待用户点击计算”和“立即执行计算” |
| `forwardRequestSeq` | 防止旧请求晚返回后覆盖新结果 |
| `forwardInFlight` / `forwardRerunRequested` | 如果计算中又修改参数，当前请求结束后再跑一次 |
| `forwardDebounceTimer` | 避免短时间连续触发造成请求过多 |

这说明你不是简单地“点按钮发请求”，而是处理了异步请求竞争和用户连续修改参数的场景。

### 5.6 读右侧检查器和公式说明

重点看：

```text
selectedForwardResult
selectedLayerFormula
buildLayerFormula()
validationIssues
fieldIssueMap
layerErrors
```

右侧面板不是自己重新计算，而是从 `forwardResult.layerResults` 里取当前选中层的结果。

每层公式说明由 `buildLayerFormula()` 根据层类型生成，例如：

```text
conv2d    卷积公式、输出尺寸公式、参数摘要
pool2d    池化尺寸变化
flatten   展平成向量
dense     矩阵乘法 + bias + activation
activation 激活函数
dropout   训练/推理状态下的差异
```

错误提示来自 Python 返回的 `validationIssues`。前端把它整理成按层、按字段的映射，用于高亮具体参数输入框。

面试时可以这样讲：

> 检查器的职责是解释后端结果，而不是自己再算一遍。这样保证页面展示的 shape、tensor、错误提示都和真实 forward 结果一致。

### 5.7 读特征图和通道预览

重点看：

```text
tensorToImageDataUrl()
grayValuesToImageDataUrl()
buildChannelPreviews()
sampleTensorForPreview()
channelImageUrl()
```

这里解决的是性能问题。早期如果把 tensor 的每个像素都渲染成 DOM 小块，图片稍大就会卡。现在的策略是：

```text
tensor values
  -> Canvas ImageData
  -> data URL
  -> <img> 展示
```

多通道特征图默认只展示部分通道，完整通道放到弹窗里。这样既能看结果，又不会让页面一次性渲染过多 DOM。

这是一个很适合面试讲的优化点：

> 特征图是数值数据，但页面展示阶段把它转换为 Canvas 图片，减少 DOM 节点数量，提升滚动和参数调整时的响应速度。

### 5.8 读卷积核对比功能

重点看：

```text
KERNEL_PRESETS
runKernelCompare()
layersForKernelCompare()
ensureConvKernelBank()
```

这个功能的思路是：针对当前选中的卷积层，把不同预设卷积核临时替换进去，然后分别请求后端 forward，比较同一输入下不同 kernel 的输出效果。

流程是：

```text
选择卷积层
  -> 遍历 KERNEL_PRESETS
  -> layersForKernelCompare() 克隆当前网络并替换当前层 kernel
  -> executeForward()
  -> 找到该卷积层输出
  -> 生成预览图和统计值
```

注意这里没有污染当前主网络，而是对 `layers` 做结构化克隆后再修改。这是为了让“对比实验”和“当前真实网络状态”分离。

## 6. 第五站：理解 Python forward 服务

先看 `backend/python-forward/app.py`。

它只有两个接口：

```text
GET  /api/health
POST /api/forward
```

真正逻辑在 `backend/python-forward/forward_engine.py` 的 `execute_forward_graph()`。

### 6.1 总入口

`execute_forward_graph()` 的主流程是：

```text
接收 layers / connections / inputTensor
  -> build_execution_graph()
  -> topological_sort()
  -> 按执行顺序遍历每一层
  -> validate_layer_params()
  -> execute_operator()
  -> compute_tensor_stats()
  -> build_layer_visualization()
  -> 汇总 ForwardPassResult
```

返回结果会包含：

```text
executionOrder
layerResults
layerShapeMap
finalTensor
finalTopK
validationIssues
shapePath
errors
warnings
resolvedLayers
```

这和前端 `ForwardPassResult` 类型一一对应。

### 6.2 为什么要构建执行图

重点看：

```text
build_execution_graph()
topological_sort()
```

虽然 A 模式前端主要是线性网络，但后端仍然按图来执行。这样有两个好处：

1. 可以检查连接是否合法。
2. 后续如果扩展多输入层、残差连接等结构，不需要推翻整个执行框架。

### 6.3 参数校验

重点看：

```text
validate_layer_params()
infer_layer_output_shape()
```

例如：

```text
Conv/Pool 要求输入是 [H, W, C]
kernel/stride/padding/dilation 组合不能导致输出尺寸小于等于 0
Dense/Output 要求输入 shape 非空
Dropout rate 必须在 [0, 1)
Residual 要检查主分支和 shortcut shape 是否一致
```

校验问题不会只在后端日志里消失，而是以 `validationIssues` 返回给前端，让页面能定位到具体层和字段。

### 6.4 各层 operator

重点看：

```text
run_input_operator()
run_conv2d_operator()
run_pool2d_operator()
run_flatten_operator()
run_dense_operator()
run_activation_operator()
run_dropout_operator()
run_output_operator()
```

这里是“真实前向传播”的核心。

卷积层：

```text
输入 reshape 成 [H, W, C]
根据 kernelSize / stride / padding / dilation 计算输出尺寸
resolve_kernel_3d() 生成每个输出通道、输入通道的 kernel
执行卷积
应用 activation
返回 tensor + transitionNote + paramsSummary
```

池化层：

```text
按窗口滑动
max 或 avg
保持通道数不变
```

Flatten：

```text
把所有 values 拉平成一维向量
```

Dense / Output：

```text
dense_weight_matrix() 取得权重
输入向量乘权重矩阵 + bias
应用 activation 或 softmax
```

如果前端没有手动传完整权重，Python 会基于层 id 生成确定性的 synthetic weight。这样同一网络重复运行结果稳定，适合教学演示。

### 6.5 可视化摘要和统计值

重点看：

```text
build_layer_visualization()
compute_tensor_stats()
downsample_tensor3d()
```

Python 不会把所有可视化工作都交给前端。它会为每层结果生成轻量摘要：

```text
3D tensor -> image visualization + channel previews
vector -> vector visualization
stats -> min/max/mean/nonZeroRatio/topK
```

前端右侧检查器拿到这些数据后，负责把它渲染成图、柱状条、Top-K 列表和 shape 路径。

## 7. 第六站：理解 Spring forward 代理

读 `backend/spring/src/main/java/com/deepvision/studio/forward/ForwardProxyController.java`。

这个 Controller 暴露给前端的是：

```text
GET  /api/forward/health
POST /api/forward
```

它本身不做数值计算，只做代理：

```text
Angular
  -> POST /api/forward
  -> Spring ForwardProxyController
  -> RestTemplate
  -> Python /api/forward
  -> 原样返回 JSON
```

为什么要加这一层？

1. 前端只需要记住一个后端入口 `/api`。
2. Docker 内部 Python 地址和本地开发地址不同，由 Spring 配置项屏蔽差异。
3. 可以统一设置超时和错误处理。
4. 后续如果加鉴权、日志、限流，不需要改前端。

配置项在 `backend/spring/src/main/resources/application.yml` 和 `docker-compose.yml` 中：

```text
deepvision.forward.base-url
DEEPVISION_FORWARD_BASE_URL
```

本地可以是 `http://127.0.0.1:5000`，Docker 中是 `http://python-forward:5000`。

## 8. 第七站：理解登录注册和 JWT

前端入口：

```text
frontend/src/app/core/auth/auth.service.ts
frontend/src/app/core/api/api-client.service.ts
frontend/src/app/core/auth/auth-page.component.ts
```

后端入口：

```text
backend/spring/src/main/java/com/deepvision/studio/auth/AuthController.java
backend/spring/src/main/java/com/deepvision/studio/auth/JwtService.java
backend/spring/src/main/java/com/deepvision/studio/auth/JwtAuthFilter.java
backend/spring/src/main/java/com/deepvision/studio/common/SecurityConfig.java
```

### 8.1 前端认证状态

`ApiClientService` 是所有前端请求的底层封装。它做了两件关键事：

```text
保存 token 到 localStorage
请求时自动加 Authorization: Bearer <token>
```

`AuthService` 再进一步维护当前用户：

```text
login()
register()
restoreSession()
logout()
user$
```

A 模式页面订阅 `user$`，用户登录后自动加载历史记录，退出后清空历史记录。

### 8.2 后端认证逻辑

`AuthController` 提供：

```text
POST /api/auth/register
POST /api/auth/login
GET  /api/auth/me
```

注册时密码用 BCrypt 存储；登录成功后 `JwtService.issue()` 签发 JWT；之后前端请求带 token，`JwtAuthFilter` 解析 token 并放入 Spring Security 上下文。

`SecurityConfig` 中对接口做了区分：

```text
公开：auth、health、forward、llm、uploads、swagger 等
需要登录：未显式放行的接口，例如 A 模式历史记录
```

面试时可以强调：A 模式 forward 本身可以公开体验，但保存历史记录必须绑定用户。

## 9. 第八站：理解 A 模式历史记录

前端：

```text
frontend/src/app/shared/forward/forward-record.models.ts
frontend/src/app/shared/forward/forward-record.service.ts
mode-a-page.component.ts 中的 saveForwardRecord/loadForwardRecords/restoreForwardRecord/deleteForwardRecord
```

后端：

```text
ForwardRecordController.java
ForwardRecord.java
ForwardRecordRepository.java
ForwardRecordDtos.java
LocalImageStorage.java
```

### 9.1 保存的不是零散字段，而是快照

前端保存时调用：

```text
buildForwardRecordSnapshot()
```

快照包含：

```text
selectedTemplateId
selectedDataset
selectedSampleId
selectedLayerId
uploadComputeProfile
uploadedImageUrl
layers
connections
forwardResult
```

也就是说，保存记录不是只存网络名称，而是存下足够恢复整个 A 模式页面的状态。

同时还会生成 `previewImageDataUrl`，用于历史记录列表显示缩略图。

### 9.2 后端如何保存

`ForwardRecordController.create()` 做了三件事：

```text
根据 Principal 找到当前用户
LocalImageStorage.saveDataUrl() 保存预览图
ObjectMapper 把 snapshot 写成 JSON 字符串
保存 ForwardRecord 实体
```

`ForwardRecord` 表里有：

```text
user
name
templateId
datasetName
layerCount
parameterCount
imagePath
snapshotJson
createdAt
```

其中 `snapshotJson` 是大字段，保存完整快照；其他字段是列表展示和查询所需的元数据。

### 9.3 如何保证只能访问自己的记录

查询和删除都使用：

```text
findByIdAndUserUsername(id, principal.getName())
findByUserUsernameOrderByCreatedAtDesc(principal.getName())
```

所以记录天然按当前登录用户隔离。

### 9.4 回溯逻辑

前端回溯时调用：

```text
restoreForwardRecord()
  -> ForwardRecordService.detail(id)
  -> applyForwardRecordDetail()
```

`applyForwardRecordDetail()` 会把 snapshot 中的模板、样本、选中层、layers、connections、forwardResult 全部恢复到页面状态。如果记录包含上传图片，还会根据后端保存的图片 URL 拉回图片并重新构建输入 asset。

这块可以这样讲：

> 历史记录不是重新跑一遍实验，而是优先恢复当时的完整页面快照；如果用户需要继续修改，再通过现有的 `runForward()` 重新计算。

## 10. 第九站：理解 3D 网络展示

A 模式入口在：

```text
mode-a-page.component.ts -> openNetwork3dViewer()
```

3D 页面在：

```text
frontend/src/app/shared/network-3d/network-3d.models.ts
frontend/src/app/shared/network-3d/network-3d-viewer.component.ts
frontend/src/app/shared/network-3d/network-3d-layout.ts
```

### 10.1 A 模式如何传数据给 3D 页面

`openNetwork3dViewer()` 会构造一个 `Network3dPayload`：

```text
layers
connections
shapeHints
layerShapes
layerSnapshots
shapePath
finalTopK
inputImageUrl
selectedLayerId
```

然后写入：

```text
sessionStorage[NETWORK_3D_SESSION_KEY]
```

最后打开 `/network-3d` 新页面。

这里没有通过 URL 参数传大 JSON，是因为网络层、特征图、图片 data URL 数据量较大，放 URL 不合适。

### 10.2 3D 页面如何展示

`Network3dViewerComponent.ngOnInit()` 从 sessionStorage 读取 payload，然后根据层和 shape 计算每层的 3D 布局。

页面主要做：

```text
Three.js scene/camera/renderer 初始化
根据 layerViews 创建几何体
根据 connections 画连接线
根据 layerSnapshots 显示特征图快照
支持选中层、hover、高亮、播放传播粒子
右侧显示当前层详情和 shape path
```

面试讲法：

> 3D 页面不重新计算网络，它只消费 A 模式传来的快照。这样 2D 工作台是数据生产者，3D 页面是可视化消费者，两者职责分离。

## 11. 第十站：理解 LLM 浮窗

前端：

```text
frontend/src/app/shared/llm/llm-floating-assistant.component.ts
frontend/src/app/shared/llm/llm-chat.service.ts
frontend/src/app/shared/llm/llm-prompts.ts
mode-a-page.component.ts 中的 LLM contextProvider
```

后端：

```text
backend/spring/src/main/java/com/deepvision/studio/llm/LlmController.java
backend/spring/src/main/java/com/deepvision/studio/llm/LlmChatClient.java
backend/spring/src/main/java/com/deepvision/studio/llm/LlmDtos.java
```

### 11.1 前端浮窗逻辑

`LlmFloatingAssistantComponent` 是一个可复用浮窗组件。它支持：

```text
展开/收起
普通聊天
勾选“传入页面上下文”
快捷问题
Markdown 简单渲染
SSE 流式响应
```

A 模式把自己的上下文通过 `contextProvider` 传给浮窗。上下文包括：

```text
网络层数、参数量
当前选中层
layers + params
shapePath
当前层输入/输出 tensor 摘要
输入图像或输出图像 data URL
```

所以 LLM 回答不是凭空聊天，而是能结合当前实验状态解释。

### 11.2 后端代理逻辑

`LlmController` 提供：

```text
POST /api/llm/chat
POST /api/llm/chat/stream
```

流式接口用 `SseEmitter` 返回事件：

```text
delta
done
error
```

`LlmChatClient` 负责把前端请求转换为火山方舟兼容的 `/chat/completions` 请求。API Key 只在后端环境变量中配置，不暴露给浏览器。

面试时可以强调：

> LLM 是通过 Spring 后端代理的，前端只传问题和上下文，不持有第三方模型的 API Key。流式输出用 SSE，让用户能看到逐步生成的回答。

## 12. 第十一站：理解教学帮助浮标

相关代码：

```text
frontend/src/app/shared/teaching/teaching-term.directive.ts
frontend/src/app/shared/teaching/teaching-search-fab.component.ts
frontend/src/app/shared/teaching/teaching-search.service.ts
frontend/src/app/shared/teaching/teaching-glossary.ts
frontend/src/app/shell/teaching/teaching-doc-page.component.ts
```

它的核心思路是：

```text
页面里的重要术语用 directive 标注
浮标组件提供搜索和跳转入口
glossary 维护术语解释
教学文档页承载更完整的说明
```

这个功能不改变 A 模式业务状态，只是附加一层学习辅助。面试时可以把它作为“降低教学平台使用门槛”的辅助功能来讲。

## 13. 第十二站：理解 AI 博物馆

如果老师问成员 A 除 A 模式外还做了什么，可以讲 AI 博物馆。

前端入口：

```text
frontend/src/app/modes/ai-museum/ai-museum-page.component.ts
```

后端入口：

```text
backend/spring/src/main/java/com/deepvision/studio/museum/MuseumPresenceHandler.java
```

### 13.1 前端 3D 展厅

`AiMuseumPageComponent` 使用 Three.js 和 `PointerLockControls` 实现第一人称漫游。

核心结构是：

```text
exhibits 数组定义展品
initScene() 初始化场景、相机、灯光、控制器
根据 exhibits 创建展墙、展牌和动态 artifact
animate() 每帧更新移动、动画对象、最近展品提示和在线 avatar
```

展品内容不是散落在 DOM 中，而是结构化的 `MuseumExhibit[]`，每个展品包含年份、标题、描述、标签、颜色主题和 3D 位置。这样新增展品主要是扩展数据，而不是复制页面结构。

### 13.2 WebSocket 在线状态

前端会连接：

```text
ws://host/api/museum/presence
```

后端 `MuseumPresenceHandler` 处理：

```text
afterConnectionEstablished()
handleTextMessage()
afterConnectionClosed()
```

消息类型主要有：

```text
welcome  当前用户进入房间
join     其他用户加入
pose     同步位置和朝向
leave    用户离开
```

后端维护 room 和 participant，前端把其他参与者渲染成远程 avatar。

面试讲法：

> AI 博物馆是一个 3D 展示型页面，展品由结构化数据驱动；在线参观不是用数据库持久化，而是用 WebSocket 在内存中维护房间状态并广播位置。

## 14. 第十三站：理解部署关系

看：

```text
docker-compose.yml
frontend/nginx.conf
backend/spring/src/main/resources/application.yml
```

Docker 部署拆成三个服务：

```text
frontend        Angular build 后由 Nginx 托管
spring-backend 统一 API、认证、记录、LLM、forward 代理
python-forward Flask + NumPy 执行真实 forward
```

Nginx 做两类事情：

```text
前端路由 fallback 到 index.html
/api、/uploads、/swagger-ui、/v3/api-docs 等代理到 Spring
```

Spring 通过环境变量连接其他资源：

```text
DEEPVISION_FORWARD_BASE_URL=http://python-forward:5000
DEEPVISION_DB_URL=jdbc:h2:file:/app/data/deepvision
DEEPVISION_UPLOAD_ROOT=/app/uploads
ARK_API_KEY=...
```

数据卷保存：

```text
H2 数据库
上传图片
数据集
训练任务输出
```

即使面试重点是 A 模式，也建议能讲清楚这条部署链，因为它体现你理解前后端分离和容器内服务名寻址。

## 15. 按功能准备面试讲解

下面是几段可以直接转换成口头回答的逻辑。

### 15.1 如果老师问：A 模式整体怎么实现？

可以这样答：

> A 模式本质是一个可编辑的神经网络前向传播实验台。前端用 `layers` 和 `connections` 表示网络结构，用 `ForwardTensor` 表示输入图像张量。用户选择模板、修改层参数或上传图片后，页面把当前网络和预处理后的输入 tensor 通过 `/api/forward` 发给 Spring。Spring 不做计算，只把请求代理到 Python forward 服务。Python 根据图结构拓扑排序，逐层执行 Conv、Pool、Flatten、Dense、Activation 等 operator，并返回每层输出 tensor、shape、统计信息、可视化摘要和校验问题。前端再用这些结果渲染右侧检查器、特征图、Top-K、shape path 和 3D 快照。

### 15.2 如果老师问：为什么要拆 Spring 和 Python？

可以这样答：

> Spring 更适合做统一 API、鉴权、数据库、Swagger 和部署配置；Python/NumPy 更适合做张量计算。前端只访问 Spring，可以避免直接暴露 Python 服务，也方便 Docker 中用不同服务名配置 forward 地址。这样计算职责、业务职责和展示职责比较清楚。

### 15.3 如果老师问：保存历史记录怎么做？

可以这样答：

> 保存记录时，前端不是只保存一个网络名称，而是构造完整 snapshot，包括模板、数据集、选中层、layers、connections 和当时的 forwardResult。同时把当前输入预览图转成 Data URL。后端根据 JWT 找到当前用户，把预览图落盘，把 snapshot 序列化成 JSON 存入 H2 的 `forward_records` 表。读取和删除记录时都按当前用户名查询，所以用户之间记录隔离。回溯时前端把 snapshot 恢复到页面状态，必要时再拉取保存的图片重建输入。

### 15.4 如果老师问：如何处理 forward 请求竞争？

可以这样答：

> A 模式里用户可能连续修改参数，所以 `runForward()` 里用了请求序号 `forwardRequestSeq` 防止旧响应覆盖新响应；用 `forwardInFlight` 和 `forwardRerunRequested` 表示计算中又发生了修改，当前请求结束后再自动跑一次；同时用 debounce 控制请求频率。这比每次输入都直接发请求更稳。

### 15.5 如果老师问：特征图展示为什么不卡？

可以这样答：

> 特征图底层是 tensor values。展示时没有把每个像素渲染成 HTML 小格子，而是用 Canvas/ImageData 一次性生成图片 data URL，再交给 `<img>` 展示。多通道 tensor 默认只展示部分通道，完整通道放在弹窗里。这样 DOM 节点数量少，页面滚动和参数调整更流畅。

### 15.6 如果老师问：3D 网络展示如何拿到数据？

可以这样答：

> A 模式打开 3D 页面时，会把当前 layers、connections、shape、每层 snapshot、输入图和 Top-K 封装为 `Network3dPayload`，写进 sessionStorage，然后打开 `/network-3d`。3D 页面只读取这个 payload 并渲染，不重新计算 forward。这样 A 模式负责产生真实实验结果，3D 页面只负责可视化。

### 15.7 如果老师问：LLM 助手怎么结合当前页面？

可以这样答：

> LLM 浮窗支持普通问答，也支持勾选传入页面上下文。A 模式提供 contextProvider，把当前网络结构、选中层、shapePath、tensor 摘要和最多几张输入/输出图像传给 LLM。前端通过 `/api/llm/chat/stream` 请求 Spring，Spring 再带后端环境变量里的 API Key 请求模型服务，并用 SSE 把 delta 流式返回给前端。

## 16. 推荐阅读顺序清单

如果你时间有限，按这个顺序读最有效：

1. `frontend/src/app/app.routes.ts`
2. `frontend/src/app/shared/simulation/sim-models.ts`
3. `frontend/src/app/shared/simulation/sim-engine.ts`
4. `frontend/src/app/modes/mode-a/mode-a-page.component.ts` 中的状态字段和 `ngOnInit()`
5. `mode-a-page.component.ts` 中的 `applyTemplate()`、网络编辑、样本选择、`runForward()`
6. `frontend/src/app/shared/forward/forward-backend.service.ts`
7. `backend/spring/src/main/java/com/deepvision/studio/forward/ForwardProxyController.java`
8. `backend/python-forward/app.py`
9. `backend/python-forward/forward_engine.py` 的 `execute_forward_graph()` 和各层 operator
10. `mode-a-page.component.ts` 中的 `buildLayerFormula()`、特征图转换、卷积核对比
11. `frontend/src/app/shared/forward/forward-record.models.ts` 和 `forward-record.service.ts`
12. `backend/spring/src/main/java/com/deepvision/studio/forward/ForwardRecordController.java`
13. `backend/spring/src/main/java/com/deepvision/studio/auth` 和 `common/SecurityConfig.java`
14. `frontend/src/app/shared/network-3d`
15. `frontend/src/app/shared/llm` 和 `backend/spring/src/main/java/com/deepvision/studio/llm`
16. `frontend/src/app/modes/ai-museum` 和 `backend/spring/src/main/java/com/deepvision/studio/museum`
17. `docker-compose.yml` 和 `frontend/nginx.conf`

## 17. 最后用一张数据流图收束

把 A 模式理解成下面这条主线即可：

```text
路由 /mode-a
  -> ModeAPageComponent 初始化模板和样本
  -> layers + connections 表示网络
  -> currentInputAsset.prepared.tensor 表示输入
  -> runForward()
  -> ForwardBackendService POST /api/forward
  -> Spring ForwardProxyController
  -> Python execute_forward_graph()
  -> ForwardPassResult
  -> A 页面右侧检查器 / 特征图 / Top-K / 错误提示
  -> 可选：保存为 ForwardRecord
  -> 可选：传给 Network3dViewer
  -> 可选：作为 LLM 上下文解释当前实验
```

面试时不要陷入“我写了很多组件”的描述，而要围绕这条数据流讲：网络结构如何表示，输入如何变成 tensor，真实 forward 在哪里执行，结果如何回到页面，保存和 3D/LLM 如何复用同一份结果。这就是成员 A 代码的主线。
