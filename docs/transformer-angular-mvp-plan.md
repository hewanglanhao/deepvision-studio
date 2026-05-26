# Transformer Explainer Angular 最低限度重构计划

## 1. 目标

本计划面向 `D:\VS Code\transformer-explainer` 项目，目标不是完整复刻原版 Svelte 应用，而是在当前平台中以 **最低限度可演示** 的方式，将其重构为一个 Angular 原生模块。

最低限度的含义是：

- 不追求原版所有交互和动画全部迁移。
- 只保留足够支撑教学演示和答辩说明的一条核心功能链。
- 页面风格必须与现有 `Mode A / B / C / D / E` 保持统一。
- 登录状态、AI 助手、问号教学入口必须和其他模式一样接入。

---

## 2. 项目现状评估

### 2.1 原项目技术特征

`transformer-explainer` 当前是一个完整的 SvelteKit 应用，而不是组件库。

核心特征：

1. 使用 `SvelteKit + Vite + Tailwind + Sass`。
2. 浏览器端直接运行 `GPT-2 ONNX` 模型。
3. 使用：
   - `onnxruntime-web`
   - `@xenova/transformers`
   - `d3`
   - `gsap`
4. 页面主入口是：
   - `src/routes/+page.svelte`
5. 大量状态集中在：
   - `src/store/index.ts`
6. 关键可视化组件包括：
   - `Attention.svelte`
   - `AttentionMatrix.svelte`
   - `Embedding.svelte`
   - `LinearSoftmax.svelte`
   - `QKV.svelte`
   - `Sankey.svelte`
   - `SubsequentBlocks.svelte`
   - `Textbook.svelte`

### 2.2 结论

这个项目和 `cnn-explainer` 一样，不适合“直接作为组件搬进 Angular”。  
真正可复用的是：

- 静态模型资源
- 推理流程
- 示例输入
- 教学内容组织方式
- 可视化思路

不适合直接复用的是：

- Svelte 组件本身
- 全局 store 结构
- 页面级动画和布局系统
- 整页级 D3 / GSAP 交互

因此，这项工作应定义为：

**在 Angular 中重建一个 Transformer Explainer 的 MVP 解释器，而不是迁移整个 Svelte 应用。**

---

## 3. 最低限度 MVP 范围

### 3.1 必须做的核心功能

最低限度版本建议只实现以下 4 个功能块：

1. 输入文本与示例切换
2. 下一词预测 Top-K 概率
3. 单层、单头 Attention Matrix 可视化
4. 一块固定结构的解释报告/教学说明区

### 3.2 暂不纳入 MVP 的部分

以下功能不进入最低限度版本：

1. 完整 Sankey 主画布
2. Block transition 动画
3. 多阶段展开式交互
4. 全量真实数值版 QKV / MLP / Subsequent Blocks 分步解释
5. Weight popovers
6. 全文 Textbook 教材系统
7. 原版所有动画与精细视觉过渡

### 3.3 QKV 分阶段策略

QKV 不建议在 MVP 第一阶段就做成“真实数值全链路演示”。

原因：

1. 当前原项目中真正稳定接入运行时数据的是 logits 和 attention。
2. QKV 可视化虽然存在，但其中一部分更偏教学示意，而不是完整真实张量驱动。
3. 如果一开始就追求真实 Q / K / V 数值、权重、bias、head 切片和逐维展示，会显著抬高重构成本。

因此建议采用两段式策略：

1. MVP 第一阶段先完成：
   - 输入文本
   - Top-K 预测
   - Attention Matrix
   - 解释报告
2. MVP 后续增强阶段再补：
   - 教学演示版 QKV
   - 与当前 token / block / head 联动
   - 用流程说明和局部交互解释 Q、K、V 的角色

这里的“教学演示版 QKV”指：

- 强调 Query / Key / Value 的作用关系
- 可以展示 token 向量进入 QKV 投影后的结构变化
- 可以与当前 head 选择联动
- 不要求第一版就逐项展示真实模型中的全部 Q / K / V 张量值

这样做的好处是：

1. 不会阻塞 MVP 主链落地。
2. 后续补 QKV 时可以直接复用已经完成的：
   - tokenizer
   - ONNX 推理
   - token 状态
   - block / head 状态
   - 平台统一页面壳和共享模块
3. 如果后续时间允许，再把教学版 QKV 逐步升级为更真实的数值版，也不会推倒重来。

### 3.4 MVP 演示闭环

最低限度版本完成后，用户应能完成如下演示流程：

1. 选择一个预置文本样例
2. 修改输入文本
3. 运行模型推理
4. 查看下一词 Top-K 概率
5. 选择一个 block 和一个 attention head
6. 查看对应 attention matrix
7. 查看自动生成的解释总结

只要这条链路打通，就已经满足“最低限度复现一部分功能”的要求。

---

## 4. 与平台统一的要求

本次重构不只是功能接入，还必须遵守你当前项目的页面和架构规范。

### 4.1 页面风格统一要求

新模块必须和 `Mode A / B / C / D / E` 保持统一，具体要求：

1. 使用平台统一顶栏壳
2. 保持卡片、按钮、输入控件、面板阴影、圆角、标题层级一致
3. 不保留 Svelte 原版那种完全独立的页面视觉语言
4. 保持和平台当前“实验模块页”的同一套交互感觉

建议直接复用现有模式页的视觉体系：

- 顶栏：参考 `Mode A` 的 `app-platform-topbar`
- 容器：参考 `Mode C` 的 `panel-card / detail-card / context-card`
- 概率列表：可参考 `Mode C` / `Mode A` 当前已有样式

### 4.2 登录状态要求

新模块必须和其他模式一样显示平台登录状态。

要求：

1. 顶栏中显示当前用户状态
2. 支持跳转登录/注册页
3. 支持退出登录
4. 不在模块内部单独实现认证系统

接入方式：

- 复用平台现有页面壳与认证状态逻辑
- 由 mode page 或 shell 提供 user / logoutRequested

### 4.3 AI 模块要求

新模块必须像 `Mode A`、`Mode C` 一样接入共享 AI 助手。

要求：

1. 复用 `shared/llm/` 下的共享浮动助手组件
2. 接法与 `Mode A / Mode C` 一致
3. Transformer 模块只提供：
   - system prompt
   - quick prompts
   - context provider

最低限度上下文建议包含：

- 当前输入文本
- 当前 token 列表
- 当前预测 top-k
- 当前 block
- 当前 attention head
- 当前 attention matrix 摘要

### 4.4 问号教学入口要求

新模块必须像 `Mode A`、`Mode C` 一样接入共享教学入口。

要求：

1. 挂载 `app-teaching-search-fab`
2. 对关键术语加 `appTeachingTerm`
3. 后续可补 Transformer 专属 glossary 词条

最低限度先标注这些术语：

- `transformer`
- `token`
- `embedding`
- `attention`
- `attention-head`
- `query-key-value`
- `softmax`
- `top-k`
- `logit`

---

## 5. 推荐的最低限度 Angular 方案

### 5.1 模块定位

建议将该模块作为一个新的原生模式页实现，例如：

- `Mode T`

或者如需沿用当前编号体系，也可以接到新的空模式页里，但本计划文档统一称之为：

- `Transformer Mode`

### 5.2 推荐目录结构

建议结构：

```text
frontend/src/app/modes/mode-t/
  mode-t-page.component.ts
  mode-t-page.component.html
  mode-t-page.component.css
  explainer/
    components/
      shell/
      input-panel/
      topk-panel/
      attention-matrix/
      report-panel/
    services/
      mode-t-state.service.ts
      mode-t-inference.service.ts
      mode-t-assets.service.ts
    models/
      mode-t.types.ts
```

静态资源建议放：

```text
frontend/public/mode-t/
  model-v2/
  article_assets/
  preview/
```

说明：

1. 不直接依赖外部仓库运行时路径。
2. 模型分片和示例资源必须同步到主项目。
3. 所有解释逻辑保持在 `modes/mode-t` 私有域内，不外溢到 `shared`。

---

## 6. 最低限度功能设计

### 6.1 输入与示例区

保留：

1. 一组预置文本样例
2. 输入文本框
3. 运行按钮
4. 模型加载状态提示

可复用来源：

- `src/store/index.ts` 中的 `inputTextExample`
- `src/constants/examples`

最低限度 UI：

- 左侧卡片或顶部输入栏
- 示例 chips
- 文本输入框
- “开始分析”按钮

### 6.2 Top-K 输出区

保留：

1. 下一词预测 Top-5 或 Top-10
2. token 文本
3. 概率条

最低限度 UI：

- 一个列表卡片
- 每行：
  - 排名
  - token
  - 概率条
  - 概率百分比

说明：

这块可以直接类比 `Mode C` 的 softmax 概率列表来做，样式最容易统一。

### 6.3 Attention Matrix

保留：

1. block 选择
2. head 选择
3. 对应的 attention matrix 热力图
4. 行列 token 标签

最低限度 UI：

- 一个矩阵热力图面板
- 上方两个 selector：
  - block
  - head

最低限度实现建议：

1. 第一版直接用 Angular + CSS Grid / SVG / Canvas 实现
2. 不强依赖 D3
3. 只画一个 head 的注意力热图

### 6.4 自动解释报告

最低限度解释报告区建议固定包含：

1. 当前输入文本
2. 当前预测 top-5
3. 当前 block / head
4. 当前 attention 行为摘要
5. 一段自动生成的中文解释

示例自动解释内容：

- 当前文本的最后一个 token 主要关注了哪些前文 token
- 当前 head 更偏向局部依赖还是远距离依赖
- 当前 top-1 和 top-2 概率差距是否明显

---

## 7. 技术路线建议

### 7.1 推理层

最低限度版本继续使用浏览器端推理：

- `onnxruntime-web`
- `@xenova/transformers`

建议复用原项目思路：

1. 加载 tokenizer
2. 合并 `gpt2.onnx.part*`
3. 创建 ORT session
4. 对输入文本编码
5. 跑推理
6. 解析 logits

### 7.2 attention 数据风险

这是最低限度方案里最关键的风险点。

需要优先验证：

1. 现有 ONNX 模型输出中是否直接包含 attention
2. 如果不包含 attention，是否能通过更换导出模型或补充中间输出节点解决

如果 attention 无法直接拿到，则最低限度版本要降级为：

1. 输入文本
2. Top-K 输出
3. 自动解释报告

也就是说：

**Attention Matrix 是最低限度方案里唯一可能需要降级的功能点。**

### 7.3 动画策略

最低限度版本不做复杂 GSAP 动画。

建议：

1. 只保留轻量级过渡
2. 使用 Angular 原生动画或简单 CSS transition
3. 不迁移原项目的复杂 block 转场

---

## 8. 分阶段计划

## Phase 0：技术验证

目标：

验证 Angular 中能否跑通原项目的最小推理链。

任务：

1. 将 `model-v2` 分片同步进主项目静态资源目录
2. 在 Angular 中加载 tokenizer
3. 在 Angular 中创建 ONNX session
4. 输入一句文本，拿到 logits
5. 确认能否拿到 attention 数据

交付物：

- 一份技术可行性结论
- 决定 MVP 是：
  - `Top-K + Attention Matrix`
  - 还是 `Top-K only`

风险等级：

- 高

---

## Phase 1：原生页面壳与统一功能入口

目标：

建立一个和现有模块统一风格的 Angular 原生页面壳。

任务：

1. 新建 `Mode T` 页面
2. 接入统一顶栏
3. 接入用户状态显示与登出
4. 接入共享 AI 助手
5. 接入共享问号教学入口

验收标准：

1. 新模式页可访问
2. 风格与 `Mode A / C` 一致
3. 右下角 AI 助手和问号教学入口正常显示

风险等级：

- 低

---

## Phase 2：最小推理闭环

目标：

实现输入文本到 Top-K 预测的最低闭环。

任务：

1. 输入框与示例切换
2. 运行 GPT-2 推理
3. 显示 top-5 或 top-10 概率
4. 显示当前 token 列表

验收标准：

1. 页面能真实运行模型
2. 修改输入后预测会更新
3. Top-K 列表稳定显示

风险等级：

- 中

---

## Phase 3：Attention Matrix MVP

目标：

如果技术可行，接入单层单头 attention 可视化。

任务：

1. block selector
2. head selector
3. attention matrix 热图
4. token 轴标签

验收标准：

1. 可切换 block 和 head
2. 热图可显示
3. 能和输入 token 一一对应

风险等级：

- 高

---

## Phase 4：解释报告 MVP

目标：

形成一块固定结构的解释报告，支持演示与答辩。

报告结构建议：

1. 当前输入文本
2. Top-K 结果
3. 当前 block / head
4. attention 热图或替代说明
5. 自动生成的解释总结

验收标准：

1. 不需要额外手动拼接信息
2. 一屏内可以完成最小讲解闭环

风险等级：

- 中

---

## Phase 4.5：QKV 教学演示增强

目标：

在不引入“真实数值版 QKV 全链路”高复杂度的前提下，补上一块可演示的 QKV 教学模块。

任务：

1. 新增一个简化版 QKV 解释区
2. 展示 token 向量到 Query / Key / Value 的投影关系
3. 支持与当前 block / head 联动
4. 为 Q、K、V 各自补中文教学说明
5. 将该模块接入共享教学词条与 AI 上下文

验收标准：

1. 用户能直观看到 Q / K / V 的角色区分
2. 页面能从 Attention Matrix 继续过渡到 QKV 教学说明
3. 不依赖完整真实 QKV 张量展示，也能完成答辩级讲解

风险等级：

- 中

---

## Phase 5：教学与 AI 深化

目标：

让新模块像 `Mode C` 一样具备解释辅助能力。

任务：

1. 扩展教学 glossary
2. 为 Transformer 术语补词条
3. 为 AI 助手增加：
   - Transformer 系统提示词
   - quick prompts
   - 模块上下文

建议 quick prompts：

1. “当前 head 主要关注了哪些 token？”
2. “为什么模型更偏向这个预测词？”
3. “这个 attention pattern 说明了什么？”
4. “当前 top-1 和 top-2 的差距意味着什么？”

风险等级：

- 低

---

## 9. 与其他模块统一的具体要求

最低限度版本必须满足以下统一性要求：

### 9.1 布局统一

1. 顶栏结构复用平台统一组件
2. 页面主内容使用与 `Mode C` 接近的 `panel-card` / `detail-card` 风格
3. 控制区、输出区、说明区统一采用平台卡片化布局

### 9.2 交互统一

1. 登录态显示与退出方式与 `Mode A / C` 一致
2. AI 助手入口固定在右下角
3. 教学问号固定在右下角
4. 模块内按钮使用平台已有按钮风格，不保留 Svelte 原版 button 视觉

### 9.3 文案统一

1. 页面中文化
2. 不保留原始开发态说明
3. 不暴露：
   - prototype
   - draft
   - WIP
   - placeholder
   - TODO

---

## 10. 建议的最低限度验收标准

如果以下内容全部满足，就可以认为“最低限度 Angular 重构版”已经达标：

1. 新模块以 Angular 原生页面运行
2. 样式风格与现有模块统一
3. 顶栏可显示登录状态并支持登出
4. 右下角有 AI 助手
5. 右下角有问号教学入口
6. 可输入文本并运行 Transformer 推理
7. 可展示 Top-K 预测
8. 如果技术允许，可展示 Attention Matrix
9. 有一块固定结构的解释报告区

---

## 11. 最终建议

### 建议结论

这项工作**适合做“最低限度 MVP 重构”**，但不适合一开始追求原版完整复刻。

### 最优落地策略

建议按这个顺序推进：

1. 先做技术验证
2. 先跑通推理
3. 再做最小可视化
4. 最后接入 AI 和教学

### 关键判断点

这个项目最低限度重构是否顺利，核心只取决于一个点：

**Angular 中能否稳定拿到 attention 数据。**

如果可以，就做：

- 输入 + Top-K + Attention Matrix + 报告

如果不可以，就先做：

- 输入 + Top-K + 报告 + AI + 教学入口

这依然是一个合格的最低限度版本。

---

## 12. 推荐下一步

下一步建议直接进入：

**Phase 0 技术验证**

优先验证：

1. Angular 中加载 GPT-2 ONNX 是否稳定
2. 是否能拿到 logits
3. 是否能拿到 attention

只有这一步结论明确后，后面的实施成本和版本边界才能真正收紧。
