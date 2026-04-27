# AI 工具使用记录

本文件用于记录项目中由 AI 协助生成或修改的内容，便于课程答辩和规范审查。

## 2026-04-12 前端原型

- 工具：GitHub Copilot / Codex
- 范围：前端原型搭建与界面交互逻辑
- 主要文件：
  - `frontend/src/app/app.ts`
  - `frontend/src/app/app.html`
  - `frontend/src/app/app.css`
  - `frontend/src/styles.css`
  - `README.md`

## 代表性提示词意图

- 从零开始构建一个深度学习可视化仿真平台前端。
- 参考 TensorFlow Playground 与 CNN Explainer 的交互设计。
- 使用 Angular 实现可演示功能。
- 包含网络结构编辑、数据集预览、训练控制、训练曲线、单样本推理、结构影响实验、预设任务评估和可解释性可视化。

## 人工复核建议

- 对训练模拟公式进行人工调整，使行为更贴合课程实验数据。
- 将模拟评估结果替换为真实后端返回指标。
- 在提交或答辩前复核 UI 文案、参数范围和接口异常处理。
