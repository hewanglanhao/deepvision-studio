# Mode D 资源安装与验证

Mode D 的 Angular 源码保存在仓库中，约 626 MiB 的 GPT-2 ONNX 模型不进入 Git 历史。模型来源于
[Transformer Explainer 官方站点](https://poloclub.github.io/transformer-explainer/)，并保留其
[MIT License](https://github.com/poloclub/transformer-explainer/blob/main/LICENSE)。

## 本地启动

```bash
cd frontend
npm ci
npm run setup:mode-d
npm start
```

安装器会执行四件事：

1. 从官方站点下载 `gpt2.onnx.part0` 到 `gpt2.onnx.part62`；
2. 逐个校验分片大小和 Git blob SHA-1，损坏或版本变化会直接失败；
3. 从锁定的 npm 依赖复制 ONNX Runtime Web 和 Transformers.js 浏览器运行时；
4. 下载固定版本 `Xenova/gpt2` tokenizer，并写入本地资源清单。

已经下载的有效文件会被复用。中途失败后重新执行同一命令即可。

## 只做完整性检查

```bash
cd frontend
npm run check:mode-d
```

检查会读取约 626 MiB 模型文件，因此可能需要几十秒。

## 可选参数

直接运行 Node 安装器时可以控制并发数或资源镜像：

```bash
node scripts/setup-mode-d-assets.mjs --concurrency 2
node scripts/setup-mode-d-assets.mjs --force
node scripts/setup-mode-d-assets.mjs --source-base https://your-mirror.example/mode-d
```

Windows 用户也可以从仓库根目录执行兼容入口：

```powershell
.\scripts\sync-transformer-explainer.ps1
.\scripts\sync-transformer-explainer.ps1 -Check
```

## Docker

`frontend/Dockerfile` 会在镜像构建阶段自动安装 Mode D 资源，因此全新克隆后可以直接执行：

```bash
docker compose up --build
```

首次构建需要额外下载约 626 MiB；后续未改变依赖和安装器时可复用 Docker 层缓存。

## 页面中的真实推理与降级展示

页面会明确显示当前数据来源：

- **真实 GPT-2 推理**：Tokenizer、ONNX 模型和注意力输出均已加载；
- **教学降级数据**：资源缺失或加载失败，页面保留交互，但 Top-K 和注意力不是模型输出。

这避免了页面可交互时被误认为真实模型已经成功初始化。
