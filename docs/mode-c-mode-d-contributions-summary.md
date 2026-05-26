# Mode C 与 Mode D 工作贡献汇总

本文档汇总本轮开发中 `Mode C` 与 `Mode D` 的主要建设成果，便于用于阶段汇报、论文撰写、答辩准备和后续交接。

## 一、总体说明

本轮工作围绕两个核心模式展开：

- `Mode C`：CNN 卷积过程与中间特征解释模块
- `Mode D`：Transformer 下一词预测与注意力解释模块

两者都已经从“原型/占位状态”推进到“可独立演示的解释型模块”阶段，并统一接入了平台已有的：

- 登录状态显示
- AI 助手入口
- 右下角问号教学入口
- 与平台其他模式一致的页面壳和视觉风格

---

## 二、Mode C 贡献汇总

### 2.1 模块形态重构

完成了 `cnn-explainer` 从外部项目到平台内原生模块的改造路线。

主要成果：

- 从最初的 `iframe` 承载方案，逐步过渡为 Angular 原生模块。
- 在 `frontend/src/app/modes/mode-c/` 下建立了完整的 `Mode C` 模块结构。
- 构建了清晰的分层目录：
  - `components`
  - `services`
  - `models`
- 页面壳、主视图、细节面板、状态层、推理层全部独立到 `Mode C` 内部，符合当前项目的模式化架构。

对应主要文件：

- [mode-c-page.component.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/mode-c-page.component.ts>)
- [mode-c-explainer-shell.component.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/components/shell/mode-c-explainer-shell.component.ts>)
- [mode-c.types.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/models/mode-c.types.ts>)
- [mode-c-state.service.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/services/mode-c-state.service.ts>)

### 2.2 CNN 真实推理链接入

完成了 `Mode C` 的真实模型推理接入，并修复了早期“模型描述不一致导致分类错误”的问题。

主要成果：

- 不再使用不一致的 `nn_10.json` 作为实际推理底座。
- 改为直接加载原始 `tfjs` 模型 `model.json`，保证推理结果与原始 `cnn-explainer` 一致。
- 修复了 `flatten` 顺序错误导致的输出偏差问题。
- 完成了真实样例图加载、真实中间层输出获取、真实分类结果与 softmax 排名计算。

对应主要文件：

- [mode-c-inference.service.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/services/mode-c-inference.service.ts>)
- [mode-c-assets.service.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/services/mode-c-assets.service.ts>)

### 2.3 Overview 原生重建

完成了 `Mode C` 主画布的 Angular 原生重建，并将其从单层摘要卡片逐步推进到更接近原版的拓扑解释视图。

主要成果：

- 支持样例图切换与模型前向运行。
- 支持 CNN 网络层顺序可视化。
- 支持每层通道缩略图显示。
- 支持层内多通道展开。
- 支持卷积层、激活层、池化层、输出层的主画布联动解释。
- 主画布布局已改为纵向拓扑优先，突出 explanation canvas。

对应主要文件：

- [mode-c-overview.component.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/components/overview/mode-c-overview.component.ts>)
- [mode-c-overview.component.html](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/components/overview/mode-c-overview.component.html>)
- [mode-c-overview.component.css](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/components/overview/mode-c-overview.component.css>)
- [mode-c-preview-canvas.component.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/components/preview-canvas/mode-c-preview-canvas.component.ts>)

### 2.4 卷积解释链路

完成了 `Mode C` 中最核心的卷积解释流程，形成了可展示的教学闭环。

主要成果：

- 支持单通道卷积最小过程解释：
  - 输入 patch
  - kernel 权重
  - 逐元素乘积
  - weighted sum
  - bias
  - output value
- 支持多输入通道共同汇聚到同一输出格的教学演示。
- 支持逐步解释：
  - `Source`
  - `Patch × kernel`
  - `Products`
  - `Accumulate`
- 支持局部格子 hover 联动。
- 支持中间输出格、通道贡献、累计过程与目标输出格回流可视化。
- 修正了“一个 3×3 patch 对应一片输出区域”的表述错误，回归为严格的 `3×3 -> 1×1` 解释。

### 2.5 ReLU / Pool / Output 解释

完成了卷积之后的关键中间步骤解释。

主要成果：

- `ReLU`：
  - 输入/输出图同步高亮
  - 悬停像素联动
  - `max(0, x)` 可视化表达
- `Pooling`：
  - 输入/输出对应点联动
  - `2×2` 窗口 max 比较值列出
  - 最终 max 结果显示
- `Output / Softmax`：
  - 完整 10 类概率显示
  - 纠正早期仅显示 Top-3 导致的误判

对应主要文件：

- [mode-c-detail-panel.component.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/components/detail-panels/mode-c-detail-panel.component.ts>)
- [mode-c-detail-panel.component.html](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/components/detail-panels/mode-c-detail-panel.component.html>)
- [mode-c-detail-panel.component.css](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/components/detail-panels/mode-c-detail-panel.component.css>)

### 2.6 Grad-CAM 功能

完成了 `Mode C` 的 Grad-CAM 第一版能力。

主要成果：

- 基于当前真实 `tfjs` CNN 模型计算 Grad-CAM。
- 默认解释当前预测类别，并支持切换目标类别。
- 输出包括：
  - 热力图矩阵
  - 热力图图片
  - 原图叠加图
  - 主导通道权重摘要
- 在细节面板中提供最小可用版 Grad-CAM 展示。

对应主要文件：

- [mode-c-types.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/models/mode-c.types.ts>)
- [mode-c-inference.service.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/services/mode-c-inference.service.ts>)
- [mode-c-state.service.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-c/explainer/services/mode-c-state.service.ts>)

### 2.7 可演示解释报告

按照既定要求，为 `Mode C` 组织出一版固定结构的“可演示解释报告”。

报告内容包括：

- 当前样本
- Top-5 / softmax 分类结果
- 选中卷积层通道特征图
- 选中通道的 patch / kernel / product 过程
- Grad-CAM 热力图叠加
- 自动生成的文字解释

### 2.8 AI 与教学模块接入

`Mode C` 已按与 `Mode A` 相同的方式接入共享 AI 和教学模块。

主要成果：

- 复用 `shared/llm` 共享 AI 助手模块。
- 复用 `shared/teaching` 右下角问号教学入口。
- 为 `Mode C` 补充了专属 teaching glossary 词条。
- AI 上下文包含样例、预测、层、通道、卷积过程、Grad-CAM 等内容。

### 2.9 文案与工程收尾

完成了 `Mode C` 的文案和开发态清理。

主要成果：

- 修复乱码和残留英文文案。
- 删除或改写 `planned / in-progress / fallback / 占位` 等开发态提示。
- 移除调试面板和调试文案。
- 调整布局：将细节面板移到 overview 下方，释放主画布空间。

---

## 三、Mode D 贡献汇总

### 3.1 模块来源与迁移

完成了 `transformer-explainer` 到平台内模式模块的最小可行重构，并最终以 `Mode D` 形态落地。

主要成果：

- 最初以 `Mode F` 作为独立开发沙盒进行开发。
- 完成开发后，移除旧 `Mode D` 内容。
- 将 `Mode F` 的 Transformer 解释器完整迁入 `frontend/src/app/modes/mode-d/`。
- 当前平台中只保留新的 `Mode D` 入口，不再暴露 `Mode F` 路由。

### 3.2 运行链打通

完成了浏览器端 Transformer 推理链的接入与验证。

主要成果：

- 本地加载：
  - ONNX 模型分片
  - `onnxruntime-web`
  - 浏览器端 transformers 运行时
- 验证了以下关键链路：
  - Tokenizer 已就绪
  - ONNX Session 已就绪
  - logits 可取到
  - attention 张量可取到
- 解决了 `.wasm / .mjs` 动态加载问题。
- 解决了运行时模块与资源路径相关问题。

对应主要文件：

- [mode-d-inference.service.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/services/mode-d-inference.service.ts>)
- [mode-d-state.service.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/services/mode-d-state.service.ts>)
- `scripts/sync-transformer-explainer.ps1`

### 3.3 Top-K 与真实 Attention Matrix

完成了 `Mode D` 的真实推理基础可视化。

主要成果：

- 输入文本后可得到真实 next-token `Top-K` 概率。
- 支持样例切换与重新推理。
- 支持单层、单头 attention matrix 可视化。
- 支持 hover / click 某个注意力单元。
- 支持当前 attention 焦点联动解释。

对应主要文件：

- [mode-d-input-panel.component.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/components/input-panel/mode-d-input-panel.component.ts>)
- [mode-d-topk-panel.component.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/components/topk-panel/mode-d-topk-panel.component.ts>)
- [mode-d-attention-matrix.component.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/components/attention-matrix/mode-d-attention-matrix.component.ts>)

### 3.4 QKV 教学可视化

完成了 `Mode D` 的教学版 QKV 可视化，而非仅停留在文字说明层。

主要成果：

- 构建了 `Query token -> Key token -> Value token -> 输出更新` 的可视化流。
- 支持 Query / Key / Score / Value / Output 五段交互。
- 支持 hover / click 锁定当前步骤。
- 顶部步骤说明浮层会跟随当前焦点变化。
- `Q / K / Value` 各自拥有局部说明卡。
- `score` 区单独补充了说明卡与 `Q × K -> 缩放 -> softmax` 小流程图。
- `Q / K / Value` 三组向量支持同维度联动高亮。
- `Value -> Output` 具备汇流路径和数值贡献可视化。
- 修复了步骤悬停时的“抖动问题”。
- 修复了 `Q × K -> 缩放 -> softmax` 三个节点文字溢出问题。

对应主要文件：

- [mode-d-qkv-panel.component.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/components/qkv-panel/mode-d-qkv-panel.component.ts>)
- [mode-d-qkv-panel.component.html](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/components/qkv-panel/mode-d-qkv-panel.component.html>)
- [mode-d-qkv-panel.component.css](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/components/qkv-panel/mode-d-qkv-panel.component.css>)

### 3.5 Transformer 解释报告

完成了 `Mode D` 的解释报告区，用于把模型当前状态组织成可演示的总结版面。

主要内容包括：

- 当前输入样例
- Top-K 下一词预测
- 当前 attention 观察
- 当前聚焦 attention 单元
- QKV 教学解释
- 自动生成的总结文本

对应主要文件：

- [mode-d-report-panel.component.ts](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/components/report-panel/mode-d-report-panel.component.ts>)
- [mode-d-report-panel.component.html](</d:/VS Code/deep-learning-plat-form/frontend/src/app/modes/mode-d/explainer/components/report-panel/mode-d-report-panel.component.html>)

### 3.6 平台能力接入

完成了 `Mode D` 与平台已有能力的统一接入。

主要成果：

- 接入登录状态显示。
- 接入 AI 助手。
- 接入右下角问号教学入口。
- 与 `Mode A / C` 保持一致的页面壳和样式语言。

### 3.7 文案汉化与收尾

完成了 `Mode D` 的人工收尾与中文化。

主要成果：

- 修复和替换可见乱码文案。
- 清理开发态提示。
- 首页入口同步更新为新的 Transformer 模块说明。
- 移除“第一阶段运行链验证”这一开发期 UI，只保留稳定成品页面。

---

## 四、跨模式共同贡献

本轮在 `Mode C` 与 `Mode D` 两条线上，还形成了以下共同型成果：

### 4.1 与平台统一的模式接入方式

- 顶栏与首页入口风格统一
- 登录状态入口统一
- AI 助手入口统一
- 教学问号入口统一

### 4.2 解释型页面组织经验

形成了两套可复用的解释型页面经验：

- `Mode C`：图像模型解释链
- `Mode D`：语言模型注意力解释链

这为后续其他模式扩展“可解释 AI 教学模块”提供了参考模板。

### 4.3 工程化沉淀

- 建立并验证了浏览器端模型推理接入策略
- 建立了外部 explainer 项目迁移到 Angular 模块的工作路径
- 完成了从“独立前端应用”到“平台模式模块”的工程落地实践

---

## 五、最终交付状态

### Mode C 当前状态

`Mode C` 已达到：

- 可独立演示
- 可解释 CNN 卷积过程
- 具备 Grad-CAM
- 具备 AI 与教学入口
- 具备解释报告

### Mode D 当前状态

`Mode D` 已达到：

- 可独立演示
- 可解释 next-token Top-K 预测
- 可解释单层单头 attention matrix
- 具备教学版 QKV 可视化
- 具备 AI 与教学入口
- 具备解释报告

---

## 六、建议用于答辩或汇报时的总结口径

可以将本轮工作概括为：

> 本轮重点完成了两个可解释教学模块的建设。  
> `Mode C` 围绕 CNN 的卷积过程、中间特征图与 Grad-CAM 展开，形成了从样本输入到卷积细节再到热力图解释的完整教学链。  
> `Mode D` 围绕 Transformer 的下一词预测、注意力矩阵与 QKV 机制展开，形成了从 Top-K 预测到注意力再到 QKV 信息流的完整演示链。  
> 同时，这两个模块都已经与平台的登录、AI 助手和教学词条体系完成统一接入，具备独立展示与持续扩展能力。

