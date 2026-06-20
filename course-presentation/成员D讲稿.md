# 成员 D 汇报讲稿

## 模式 E 引擎与子步骤动画

我负责的是模式 E 反向传播可视化和模式 F RNN 循环神经网络。

先说技术架构。两个模式都采用 Angular standalone component + 懒加载路由，每个模式的 JS chunk 约 60 KB，用户不访问就不会下载。状态管理全部用 Angular signal 和 computed，没有 RxJS BehaviorSubject 也没有 NgRx。所有状态集中在一个 `@Injectable({ providedIn: 'root' })` 单例服务中。计算引擎是纯 TypeScript 类，和 Angular 框架零耦合——拿出来在任何 JS 环境都能跑。

模式 E 做的是反向传播可视化。核心是一个手写的 MLP 训练引擎，大约 650 行纯 TypeScript，自己实现了 dot 矩阵乘法、transpose 转置、hadamard 逐元素乘和 softmax。没有依赖 NumPy、math.js 或 TensorFlow.js。

反向传播的链式法则照着课本公式逐行写的。Dense 层的 dW 等于 a_prev^T 乘以 dZ，db 等于 Σ dZ，dA_prev 等于 dZ 乘以 W^T。每种激活函数的导数分别实现——ReLU 用 (Z > 0) 做掩码，Sigmoid 是 σ(1-σ)，Tanh 是 1-tanh²。Softmax 和交叉熵做了合并梯度，dZ 直接等于 softmax 减 onehot，省一步回传。三种优化器 SGD、Momentum、Adam 的实现参数和 PyTorch 默认值一致——Momentum beta 0.9、Adam beta1 0.9 beta2 0.999 epsilon 1e-8，偏差修正也做了。

子步骤动画是 Web 层面的一个设计点。普通的训练 demo 一步直接算完，看不到数据流动。我用 TypeScript discriminated union 定义了一个 `SubStep` 状态机:

`{ type: 'forward'; layerPair: number }` → `{ type: 'loss' }` → `{ type: 'backward'; layerPair: number }` → `{ type: 'update'; layerIdx: number }`

`buildSubSteps` 方法根据当前网络层数动态生成完整序列。Overview 组件通过 computed 信号推导出每个状态下哪些连线要高亮、光点动画往哪个方向走、神经元外圈用什么颜色。全部 UI 变更由 signal 驱动，没有手动 DOM 操作。连续播放模式每 N 毫秒自动推进，到达设定步数自动保存曲线后暂停。

## 模式 E 神经元连线图与决策边界

这张图是模式 E 的主可视化——一个 SVG 神经元权重连线图。

渲染方面做了一些优化：非活动层的连线不透明度降到 0.45 减少视觉噪点；hover 交互用了 16px 宽的透明 hit area 线，实际显示线设置 `pointer-events: none`，用户不需要精确点到细线；hover 的浮层没有用 SVG text，而是用 HTML 绝对定位 div 覆盖在 SVG 上方——因为 HTML 能做 box-shadow、圆角、自动跟随鼠标，SVG text 做不了。浮层位置通过 `onSvgMove` 事件追踪鼠标在 SVG 视口内的坐标，用 `getBoundingClientRect` 转换。

功能上，连线的颜色对应不同阶段——蓝线是前向传播，橙线是反向传播，绿线是参数更新。正在计算的那对层之间的连线上还有流动光点做 animateMotion 动画，蓝点前向走、橙点反向走。点击任意神经元，详情面板列出这个神经元全部入边权重和偏置值。悬停连线弹出浮层显示权重值、梯度值或者在参数更新阶段显示变化前后的对比。

决策边界每 25 步更新一次。50 乘 50 网格需要对 2500 个点各做一次 forwardPass，在浏览器主线程上跑完大概几十毫秒。如果每步都算会拖帧，所以限制频率。结果以半透明色块叠加在左侧浮层面板的散点图上。坐标映射是根据数据集的实际最小最大值动态计算的，带 8% 边距——这是处理同心圆数据因为噪声溢出 [0,1] 范围的问题。

## 模式 E 损失曲线对比

损失曲线系统涉及较复杂的状态管理。

同时维护三种损失数据在独立的 signal 里：`lossHistory` 是每一步的单样本损失、`avgLossHistory` 是每 25 步用全量样本评估的平均损失、`savedCurves` 是历史曲线的快照数组。图表中灰色虚线是原始单步损失、实线是平滑平均损失。

训练到达设定步数后自动调用 `saveCurrentCurve` 保存。标签格式是 optimizer+activation 拼出来的，比如 Adam+Sigmoid，附上最终准确率。颜色从 8 色调色板中自动选未被使用的。同配置旧曲线会被 filter 掉替换。每条曲线旁有 x 删除按钮，调 `deleteSavedCurve(idx)` 用 immutable 方式——展开旧数组、splice、再 set 回去触发 signal 更新。

这个东西的教学价值是：同一数据集上先跑 SGD 自动保存、切换到 Adam 重置再跑、再自动保存，两条曲线叠加，收敛速度和最终 loss 的差异直接看得出来。激活函数也可以这样对比——跑一遍 Sigmoid 再跑一遍 ReLU，看曲线和决策边界的形态差异。点击图表弹出居中放大 Modal——CSS 遮罩层加白色卡片，`stopPropagation` 防止点卡片内部关闭。

## 模式 F RNN 引擎、数据集与时间展开

模式 F 和模式 E 共用同一套架构：纯 TS 引擎 + signal 状态服务 + standalone 组件。大约 180 行代码。

引擎实现了标准的 Tanh RNN：前向传播每步 h_t = tanh(W_xh·x_t + W_hh·h_{t-1} + b_h)，最后步走 softmax 分类。BPTT 在时间轴上逆序迭代——先算输出梯度、再逐步过 tanh 导数（用 1-h_t² 算 hadamard 积）、再累积三个权重矩阵的梯度、再通过 Whh^T 传到上一步。短序列 4 步在浏览器主线程上无感。

三个数据集——延迟记忆、XOR 记忆、交替检测——在服务构造时一次性生成，200 条 4 步序列，二进制输入用 one-hot 编码成 [bit, 0] 格式。

时间展开图用 SVG 渲染。每步一个 Cell 方框，方框间用带 marker-end 箭头的 line 串联。Cell 内部用 rect 条形图展示隐状态向量——蓝色是正向激活、红色是负向，高度由 abs 值决定。下方标注 softmax 输出概率。整体布局通过常量 CELL_W=130、CELL_H=80 控制，viewBox 根据时间步数和隐状态维度动态计算。

## 模式 F 详情面板与训练监控

详情面板分四个区域。权重矩阵区域用 computed 从 networkMeta 读取维度并拼接成字符串展示。梯度范数用一条 div 的宽度百分比映射 0-100%，颜色用 rgba 透明度渐变。隐状态向量逐时间步渲染条形图，每个 bar 颜色由值正负决定。输出概率区域逐时间步列出 softmax 百分比分布。

训练控制方面：训练进行中配置控件全部 disabled——`isRunning` 是 computed 从 `isPlaying` 和 `isAnimating` 两个 signal 派生。连续模式到达设定步数后 `pause()` 自动停止。指令缓存方面，优化器按钮和激活函数按钮通过 `[appTeachingTerm]` 指令绑定到教学文档对应词条，点击"?"浮标后按钮变浅绿高亮可跳转。

整个模式 F 完全在浏览器端运行，不依赖任何后端服务。和模式 E 一样复用了平台的 PlatformTopbar、AI 浮层助手和教学文档浮标——三个共享组件统一挂在 Shell 里，不需要每个模式重写一套。

以上就是我负责的模式 E 和模式 F。
