# 01 · 总览：Looped Transformers are Better at Learning Learning Algorithms

- **作者**：Liu Yang, Kangwook Lee, Robert D. Nowak, Dimitris Papailiopoulos（UW–Madison）
- **发表**：ICLR 2024 · arXiv:2311.12424v3
- **代码**：<https://github.com/Leiay/looped_transformer>
- **关键词**：in-context learning、weight sharing、input injection、fixed-point iteration、effective depth

## 一句话

在 Garg et al. (2022) 的 in-context 回归设定下，把 12 层 Transformer 换成**「1 层 + 循环 20 次 + 每次把输入加回去」**，参数从 9.48M 降到 0.79M（<10%），精度基本持平，在稀疏线性回归与决策树上反超。

## 它想回答的问题

传统 ML 解最小二乘有两种路子：

1. 闭式解 $w = (X^{\top}X)^{-1}X^{\top}y$ —— 需要矩阵求逆，Transformer 学起来很难（Charton 2021；Giannou et al. 2023 表明求逆需要很多层/头）。
2. 迭代法 $w_{t+1} = w_{t} - \eta \cdot X^{\top}(Xw_{t} - y)$ —— 只需要矩阵乘法，但**要重复很多次**。

标准 Transformer 的前向是固定的 L 层，天然不适配"重复"。所以作者问：

> **Can looped transformers emulate iterative learning algorithms more efficiently than standard, non-recursive transformers?**

答案（在测试的函数类内）：是。

## 三条贡献

1. **训练方法**：给出让循环 Transformer 学会迭代算法的一整套配方
   - 架构：$Y_{t+1} = M_\theta(Y_{t} + P)$，$Y_{0} = 0$（**输入注入**，而非权重绑定 $Y_{t+1}=M(Y_{t})$）
   - 损失：只在最后 $T$ 次循环上算（截断窗口，省显存）
   - 调度：训练中把 $b$ 从 1 线性升到目标值
2. **实验证据**：linear / sparse linear / decision tree / 2-layer ReLU NN 四类函数上，looped 匹配甚至超过 12 层标准 Transformer（尤其在稀疏与树任务上）
3. **诊断**：模型在**超过训练循环数**的迭代上仍收敛到不动点（固定点解），且 probing 显示表示在每次循环中被单调精炼

## 关键数字（务必记住这几个）

| 项目 | 标准 Transformer | Looped Transformer |
|---|---|---|
| 层数 | 12 | 1（循环 b 次） |
| 宽度 / 头数 | D=256, h=8 | D=256, h=8 |
| 参数量 | 9.48M | **0.79M**（1/12） |
| 最优 (b, T) · linear | — | (20, 15) |
| 最优 (b, T) · sparse linear | — | (20, 10) |
| 最优 (b, T) · decision tree | — | (70, 15) |
| 最优 (b, T) · 2-layer ReLU NN | — | (12, 5) |

任务设置：$d=20$，in-context 样本 $k=40$（sparse / tree / MLP 用 $k=100$），$w \sim N(0, I)$，$x \sim N(0, I)$，评估 1280 个 prompt，bootstrap 1000 次报 90% CI，误差按 Garg 的方式归一化。

稀疏线性回归上的一次具体对比（40 个 in-context 样本）：

| 解法 | 归一化误差 |
|---|---|
| OLS（最优解） | 8.21e-14 |
| Lasso（α=0.01） | 1.32e-04 |
| **Looped Transformer** | **6.12e-04** |
| Transformer (12 层) | 1.70e-03 |

注意：样本数 <10 时 looped 反而**优于** Lasso —— 作者归因于"简洁性归纳偏置"偏好稀疏解。这是全文最有意思的观察，也是最缺乏理论的观察。

## 它在文献里的位置

```
Garg 2022（TF 能 in-context 学函数类）
   ├── von Oswald 2022 / Ahn 2023 / Mahankali 2023（线性自注意力 ≈ 一步 GD）—— 理论侧，但架构被简化
   ├── Akyürek 2022（probing，揭示 TF 内部阶段的相变）
   └── 本文（标准 decoder-only TF + 循环结构 → 学迭代算法）—— 实验侧，架构贴近实践
```

技术祖宗是 **Deep Equilibrium Models / Universal Transformer**：$Y = M(Y+P)$ 的不动点就是 DEQ 的平衡点，权重共享 + 输入注入是 Bai et al. (2019) 定理 3 的形式（任意 L 层网络可由权重绑定 + 输入注入的网络表示）。本文的新意不在架构，而在**把它作为 in-context 算法学习的训练配方落地并系统评测**。

## 我的第一印象（详细批判见 04）

- 卖点"1/12 参数"是真的，但**"更便宜"是假的**：b=20 次循环的前向 ≈ 20 层的前向，比 12 层还贵。等算力比较缺失。
- 固定点性质是训练目标的直接后果（要求在 b 步内匹配），不算 emergent。
- "简洁性偏置"能解释稀疏任务变好，也能解释 x 缩放 3 倍时变差 —— 一个假设解释所有结果时，要小心它是不是不可证伪。
