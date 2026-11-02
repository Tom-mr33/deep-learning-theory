# 01 · 总览：Sparse maximal update parameterization (SµPar)

- **作者**：Nolan Dey, Shane Bergsma, Joel Hestness（Cerebras Systems）
- **arXiv**：2405.15743v3（2024 初版，v3 更新至 2026-02）
- **代码**：<https://github.com/EleutherAI/nanoGPT-mup/tree/supar>
- **关键词**：weight sparsity、μP / μTransfer、feature learning、signal propagation、HP transfer

## 一句话

把 μP 从"随**宽度**缩放"推广到"随**宽度 × 密度**缩放"，使 dense 模型上搜到的学习率与初始化方差**直接**成为任意稀疏度下的最优超参——扫稀疏度不用重调超参。

## 问题：稀疏训练的三重困境（Sec. 1）

1. **信号受损**：把大部分权重置零会破坏前向激活与梯度传播（Evci 2022；Lee 2019；Bambhaniya 2024）。
2. **调参成本爆炸**：稀疏方法本身引入额外超参（剪枝轮数等），而研究又需要在多个稀疏度上跑 → 每个稀疏度都要调 → "调参成本吃掉了稀疏省下的算力"（作者原话：defeat the purpose）。
3. **硬件生态缺失**：GPU/TPU 只有 2:4 结构化稀疏能加速；非结构化稀疏只有 Cerebras / Graphcore / SambaNova 这类硬件真正受益。

于是**标准做法**是：直接复用 dense 模型上优化好的超参。作者的实证发现（Fig.1 左）：

> **稀疏与稠密模型的最优超参并不一致**：随稀疏度提高，最优学习率与最优初始化方差会系统性地漂移（且不单调、不可预测，μP 甚至出现不连续）。

## 解法：SµPar

```
σ_W²  = σ_base² / (m_d · m_ρ)          # 隐藏层初始化方差
η_adam = η_base / (m_d · m_ρ)          # 隐藏层学习率（Adam）
η_sgd  = η_base / m_ρ                  # SGD 只差密度因子
m_d = d_model / d_model_base,  m_ρ = ρ / ρ_base
```

嵌入、解嵌入、LayerNorm、bias、注意力 logits 的缩放与 μP 保持一致（因为论文只稀疏化投影层）。

**关键性质**：$\rho = 1, m_\rho = 1$ 时退化为 μP；$\rho = 1$ 且不缩宽度时退化为标准参数化（SP）。

## 四条 Finding（这是全文骨架）

| # | 结论 | 证据 |
|---|---|---|
| 1 | 稀疏度提高 → **激活与梯度消失**（SP 与 μP 都有） | Fig.5：mean abs activation 随 density 降而崩塌（对数轴） |
| 2 | SP/μP 下，**dense 与 sparse 的最优超参不同** | Fig.6（LR）、Fig.7（init std）最优值漂移 |
| 3 | SµPar 在**任意宽度与稀疏度**下都给出稳定的激活尺度与稳定最优超参 | Fig.5/6/7/9 |
| 4 | 因此 SµPar 在**同样超参**下优于 SP 和 μP，本质是"调参更好"而非"优化器更好" | Fig.3/8/10、Table 2 |

## 关键数字

| 项目 | 数值 |
|---|---|
| 模型 | GPT 类，610M 参数（d=2048, 10 层, head 64），Chinchilla 20 token/param → 12.13B tokens |
| 数据 | SlimPajama，2048 上下文 |
| 稀疏度范围 | 0 → 99.2%（density 2⁻⁷） |
| 相对 loss 改进 | 较 SP **11.9%**，较 μP **1.9%**（99.2% 稀疏） |
| 平均（各稀疏度最优 LR 下） | 较 SP 好 0.8%，较 μP 好 2.1% |
| 下游平均准确率 @99.2% | SP 0.34 / μP 0.39 / **SµPar 0.40** / 随机 0.30 |
| 单次训练成本 | Cerebras CS-3 上 3–6 小时，≈ A100 单卡 9 天以上 |
| 调参代理模型 | d=256, 24 层, head 64 → 39M 参数，800M tokens，随机搜 350 组配置 |
| 最终超参 | σ_base = 0.08665602，η_base = 1.62e-2，α_input = 9.1705，α_output = 1.0951835 |

## 为什么这篇值得细读

1. **推导可检验**：三条 Feature Learning Desiderata（前向/反向/更新）都能用几行代码数值验证（本仓库 `tests/test_smupar.py` 就是这么做的）。
2. **"看似生效"的既有技巧被解释**：Sec. 4.2 的 **Iso-WPN** 分析说明，很多稀疏 scaling 技巧之所以 work，是因为它们让 $d_{model} \cdot \rho$（每神经元非零权重数）保持常数——这时 μP 与 SµPar 等价，SP 也能用。这是一个漂亮的"统一解释"。
3. **诚实的失败**：动态稀疏（RigL / GMP）下 SµPar **不 work**（过校正），作者明确写出来并给出机理（掩码变成非高斯，LLN 假设失效）。

## 与你已有工作的关系

- 与 **μP/μTransfer** 是直接的推广关系：把"tune small, train large"扩展为"tune dense, train sparse"。
- 与 **Looped Transformer**（本仓库另一篇）的交集：循环结构让**同一组权重被反复应用 b 次**，稀疏化造成的单次衰减会累积，SµPar 的尺度补偿是否还够？→ 见 `docs/00_reading_plan.md` 选题 2。
