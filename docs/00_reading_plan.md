# 精读总览与阅读路线

> 本目录是两篇论文的**工作笔记**，不是论文翻译。重点写：作者到底证明了什么、哪些是实验观察而非定理、哪些数字可以复现、哪里我不同意。

## 为什么把这两篇放在一起

| | Looped Transformer (LTR) | SµPar |
|---|---|---|
| 关心的问题 | **有效深度**：能不能用 1/12 的参数实现 12 层的表达力 | **信号尺度**：稀疏化破坏了什么，怎么系统修回来 |
| 手段 | 权重共享 + 输入注入 + 截断窗口损失 | 重参数化 init 与 LR（随宽度 × 稀疏度缩放） |
| 核心变量 | 循环次数 `b`、损失窗口 `T` | 宽度倍数 `m_d`、密度倍数 `m_ρ` |
| 结论类型 | 实验观察 + 固定点论证 | 方差推导（可检验）+ 大规模实验 |
| 我的评价 | 想法漂亮，比较不公平（算力 vs 参数） | 推导干净，收益有一部分是"调参"而非"方法" |

共同的更深主题：**神经网络的训练动力学可以用少数几个尺度量刻画**（激活尺度、梯度尺度、更新尺度、有效深度）。LTR 用"重复应用同一算子"换取深度；SµPar 用"补偿尺度"保住深度不变性。两者都在问：*当我们改变某个结构性超参（深度/宽度/稀疏度）时，什么滴度必须同步改变？*

## 阅读顺序

1. `looped_transformer/01_overview.md` → 背景（Garg 2022 in-context 函数类）、问题、贡献
2. `looped_transformer/02_method.md` → 公式逐项拆解（输入注入、窗口损失、b/T 选择）
3. `looped_transformer/03_experiments.md` → 逐图笔记（Fig.2–17 我实际看到了什么）
4. `looped_transformer/04_critique_and_ideas.md` → 批判 + 可做的题
5. `looped_transformer/05_reproduction_plan.md` → 与 `src/looped` 的对应关系、实验矩阵、成本
6. `smupar/01_overview.md` → 动机（为什么 sparse 不能复用 dense 超参）
7. `smupar/02_derivation.md` → App.D 三段推导，自己重推一遍（关键：ΔW 那一步）
8. `smupar/03_experiments.md` → HP 迁移、Iso-Parameter / Iso-WPN、RigL/GMP 失效
9. `smupar/04_critique_and_ideas.md`

## 前置知识清单（读不懂时回查）

- **In-context learning 作为算法学习**：Garg et al. 2022（arXiv:2208.01066）；von Oswald et al. 2022（LSA 学一步 GD）；Akyürek et al. 2022（probing）。
- **Deep Equilibrium / implicit models**：Bai et al. 2019（DEQ）——LTR 的数学祖宗，`Y = M(Y+P)` 的固定点就是 DEQ 的平衡点。
- **μP / μTransfer**：Yang & Hu 2020（arXiv:2011.14522）、Yang et al. 2021（arXiv:2203.03466）——SµPar 的直接推广对象。
- **信号传播**：fan-in/fan-out 方差分析、mean-field 视角（Poole et al. 2016）。

## 候选课题（按性价比排序）

1. **等 FLOPs 比较（低风险，能写成 workshop 论文）**
   论文只做了等参数比较。做一个"等前向 FLOPs / 等训练 FLOPs"的对照：12 层 × 1 次 vs 1 层 × b 次，其中 b 取 12（等算力）。看 LTR 是否还占优。**这是我认为最该先做的事**，因为它决定论文的核心卖点是否成立。

2. **循环次数-稀疏度交互（承接 SµPar，原创性中高）**
   把 SµPar 的缩放律用到循环架构上：loop 里同一组权重被反复使用 b 次，稀疏化会让每次应用的信号衰减累积成 b 次 → 误差随 b 指数放大还是被固定点吸收？测 `‖Y_t − Y_{t-1}‖` 随 t 与 ρ 的曲线。

3. **用循环 Transformer 学迭代反演算子（对口你的遥感水文方向）**
   把 LTR 从"学 GD 解线性回归"换成"学 4D-Var / EnKF 的数据同化迭代"，输入是观测序列 + 先验，输出是状态增量序列。LTR 的"迭代结构"正好对应同化的外循环。可与 FGFusion 的物理约束结合：把质量/能量守恒作为固定点的约束项。
   *风险*：数据与算子要自己造，工作量大，但故事最完整。

4. **固定点加速（工程向，好写代码）**
   推理时 b=100 次前向很贵。用 Anderson 加速 / 拟牛顿解 `Y = M(Y+P)` 的固定点，比较"少迭代 + 加速"与"多迭代"的精度-算力前沿。这是 DEQ 社区的标准手法，LTR 论文完全没试。

5. **probing 的因果化（方法论批判，小而锐）**
   论文用 probe 说明"每层都在精炼 X^T y"。但 probe 只能证明"信息存在"，不能证明"被使用"。做干预实验：在隐藏态里沿 X^T y 方向做投影消融 / 激活修补，看输出误差是否如预测般崩掉。

## 一句话备忘

- LTR 的"1/12 参数"是**参数**，不是**算力**；b 次循环的前向 ≈ b 层前向，b=20 > 12。
- SµPar 的收益里有一块是"把 dense 超参从欠调状态救回来"，作者自己在 Finding 4 承认了。
