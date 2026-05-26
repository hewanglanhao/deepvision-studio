# Mode D 资源同步说明

本文档说明 `Mode D` 的静态资源如何获取、同步和使用。

当前 `Mode D` 已改造成：

- **源码保留在仓库中**
- **模型与运行时大资源通过同步脚本单独获取**

这样做的目的是减小仓库体积，避免把大量模型分片、运行时二进制和图片资源长期提交进版本控制。

---

## 一、资源同步后的目录结构

`Mode D` 运行时会从下面这个目录读取静态资源：

```text
frontend/public/mode-d-assets/
```

同步完成后，目录中应至少包含：

```text
frontend/public/mode-d-assets/
├─ model-v2/
├─ article_assets/
├─ preview/
└─ vendor/
   ├─ onnxruntime/
   └─ transformers/
```

其中：

- `model-v2/`
  GPT-2 ONNX 分片
- `article_assets/`
  教学配图
- `preview/`
  预览图
- `vendor/onnxruntime/`
  ONNX Runtime 相关 `.wasm / .mjs / .js`
- `vendor/transformers/`
  浏览器端 `transformers.min.js`

---

## 二、默认同步方式

在仓库根目录执行：

```powershell
.\scripts\sync-transformer-explainer.ps1
```

默认情况下，脚本会：

1. 从外部项目目录读取资源  
   默认源目录：

```text
D:\VS Code\transformer-explainer
```

2. 同步到当前项目的前端 `public` 目录  
   默认目标目录：

```text
D:\VS Code\deep-learning-plat-form\frontend\public\mode-d-assets
```

---

## 三、脚本参数说明

同步脚本位置：

- [scripts/sync-transformer-explainer.ps1](</d:/VS Code/deep-learning-plat-form/scripts/sync-transformer-explainer.ps1>)

脚本支持三个参数：

### 1. `SourceRoot`

外部 `transformer-explainer` 项目的根目录。

默认值：

```powershell
"D:\VS Code\transformer-explainer"
```

### 2. `FrontendRoot`

当前平台项目中 `frontend` 目录的绝对路径。

默认值：

```powershell
"D:\VS Code\deep-learning-plat-form\frontend"
```

### 3. `TargetName`

同步到 `frontend/public/` 下的目标目录名。

默认值：

```powershell
"mode-d-assets"
```

---

## 四、自定义路径的用法

如果你的两个项目不在默认路径下，可以显式传参。

例如：

```powershell
.\scripts\sync-transformer-explainer.ps1 `
  -SourceRoot "E:\Projects\transformer-explainer" `
  -FrontendRoot "E:\Projects\deep-learning-plat-form\frontend" `
  -TargetName "mode-d-assets"
```

如果你只是想改目标目录名，例如同步到 `frontend/public/transformer-assets`，可以这样：

```powershell
.\scripts\sync-transformer-explainer.ps1 -TargetName "transformer-assets"
```

但要注意：

**如果改了 `TargetName`，前端代码里的资源基路径也必须同步修改。**

当前 `Mode D` 的推理服务默认读取：

```text
/mode-d-assets/...
```

对应文件：

- [mode-d-inference.service.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/services/mode-d-inference.service.ts>)

所以如果你把资源同步到了别的目录名，就需要同时修改里面的：

```ts
private readonly assetBase = '/mode-d-assets';
```

---

## 五、执行前的前提条件

执行同步脚本前，建议确认以下条件：

### 1. 外部项目存在

确认：

```text
SourceRoot\static
```

目录存在。

脚本会从这里复制：

- `model-v2`
- `article_assets`
- `preview`

### 2. 前端依赖已安装

脚本会尝试从当前前端的 `node_modules` 中复制运行时文件：

- `onnxruntime-web`
- `@xenova/transformers`

所以需要先在 `frontend/` 下安装依赖。

### 3. 目标前端项目路径正确

脚本要求 `FrontendRoot` 指向的是当前项目的：

```text
...\deep-learning-plat-form\frontend
```

而不是仓库根目录。

---

## 六、脚本实际做了什么

脚本主要分成两部分同步。

### 第一部分：复制外部静态资源

从：

```text
SourceRoot\static\
```

复制到：

```text
FrontendRoot\public\mode-d-assets\
```

包括：

- `model-v2`
- `article_assets`
- `preview`

### 第二部分：复制运行时依赖

从：

```text
FrontendRoot\node_modules\onnxruntime-web\dist
FrontendRoot\node_modules\@xenova\transformers\dist
```

复制到：

```text
FrontendRoot\public\mode-d-assets\vendor\onnxruntime
FrontendRoot\public\mode-d-assets\vendor\transformers
```

其中：

- `onnxruntime` 会同步：
  - `ort-wasm*.wasm`
  - `ort-wasm*.mjs`
  - 并额外复制一份 `.js`
- `transformers` 会同步：
  - `transformers.min.js`

---

## 七、路径相关的注意事项

### 1. 最好在仓库根目录执行脚本

推荐当前目录为：

```text
D:\VS Code\deep-learning-plat-form
```

然后执行：

```powershell
.\scripts\sync-transformer-explainer.ps1
```

这样相对路径最稳定。

### 2. `FrontendRoot` 不是仓库根目录

很多人容易把：

```text
D:\VS Code\deep-learning-plat-form
```

误传成 `FrontendRoot`。  
实际上应该传：

```text
D:\VS Code\deep-learning-plat-form\frontend
```

### 3. 修改目标目录名要同步改代码

如果：

- 脚本输出到了 `frontend/public/transformer-assets`

那么前端代码里：

```ts
private readonly assetBase = '/mode-d-assets';
```

也必须改成：

```ts
private readonly assetBase = '/transformer-assets';
```

否则页面会找错路径。

### 4. 旧目录 `mode-d/` 和 `mode-f/` 不再作为正式资源入口

当前正式资源入口是：

```text
frontend/public/mode-d-assets/
```

旧目录：

- `frontend/public/mode-d/`
- `frontend/public/mode-f/`

已经不建议继续作为正式运行目录使用。

---

## 八、同步成功后如何验证

同步完成后，可以做两类验证。

### 1. 文件目录验证

确认这些目录存在：

```text
frontend/public/mode-d-assets/model-v2
frontend/public/mode-d-assets/vendor/onnxruntime
frontend/public/mode-d-assets/vendor/transformers
```

### 2. 页面运行验证

启动前端后打开 `Mode D` 页面，检查：

- 输入文本后能否正常推理
- `Top-K` 是否显示
- `Attention Matrix` 是否显示
- `QKV` 面板是否正常

如果这些都正常，说明资源路径已经生效。

---

## 九、常见问题

### 1. 提示找不到 `transformer-explainer\static`

说明 `SourceRoot` 不对。  
检查外部项目是否真的在：

```text
D:\VS Code\transformer-explainer
```

如果不在，需要传 `-SourceRoot`。

### 2. 提示找不到 `onnxruntime-web` 或 `transformers` 分发目录

说明前端依赖还没装，或者装在了别的位置。

先在：

```text
frontend/
```

下安装依赖，再执行同步。

### 3. 页面仍然从 `/mode-d/...` 读资源

说明你当前运行的代码不是最新的，或者本地还有旧构建缓存。  
当前正确代码应读取：

```text
/mode-d-assets/...
```

### 4. 同步成功但页面仍报资源不存在

优先检查：

- `TargetName` 是否被改过
- `assetBase` 是否同步修改
- 开发服务器是否需要重启

---

## 十、建议的使用流程

推荐按下面顺序使用：

1. 拉取平台源码仓库
2. 安装 `frontend` 依赖
3. 确认外部 `transformer-explainer` 项目路径
4. 在仓库根目录执行：

```powershell
.\scripts\sync-transformer-explainer.ps1
```

5. 启动前端
6. 打开 `Mode D` 验证运行

---

## 十一、一句话总结

`Mode D` 当前采用的是：

**源码进入仓库，模型和运行时资源通过同步脚本单独获取，并统一放入 `frontend/public/mode-d-assets/`。**

