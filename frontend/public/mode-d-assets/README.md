# Mode D 外部资源目录

这个目录用于承载 `Mode D` 的外部静态资源，不建议把大模型分片和运行时二进制直接提交进仓库。

预期目录结构：

- `model-v2/`
- `article_assets/`
- `preview/`
- `vendor/onnxruntime/`
- `vendor/transformers/`

推荐使用仓库根目录下的同步脚本获取资源：

```powershell
.\scripts\sync-transformer-explainer.ps1
```

同步后，`Mode D` 会从 `/mode-d-assets/...` 读取：

- GPT-2 ONNX 分片
- ONNX Runtime wasm / mjs / js
- 浏览器端 transformers 运行时
- 教学配图与预览图

如果资源尚未同步，`Mode D` 的浏览器端推理链将无法完成初始化。

更完整的路径、参数和排查说明请查看：

- [Mode D 资源同步说明](</d:/VS Code/deep-learning-plat-form/docs/mode-d-assets-sync-guide.md>)
