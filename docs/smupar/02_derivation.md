# 02 · 推导：三条 desiderata 与三个缩放因子

> 目标：把 App. D 的推导自己走一遍，确认每一步用到的假设，并用数值实验检验（本仓库 `tests/test_smupar.py`）。

## 2.1 记号

| 符号 | 含义 |
|---|---|
| `B, d_in, d_out` | batch、输入维度、输出维度 |
| `X ∈ R^{B×d_in}`, `Y ∈ R^{B×d_out}` | 激活 |
| `W ∈ R^{d_in×d_out}`, `M ∈ {0,1}^{d_in×d_out}` | 权重、静态随机掩码 |
| `ρ = 1 − s` | 密度（1 = dense） |
| `σ_W²` | 权重初始化方差 |
| `m_d = d_model/d_model,base`, `m_ρ = ρ/ρ_base` | 宽度倍数、密度倍数 |
| `ΔW`, `ΔY = X(ΔW ⊙ M)` | 权重更新、更新对输出的影响 |
| `η` | 该层学习率 |

## 2.2 Feature Learning Desiderata（FLD）

```
‖Y_i‖₂ = Θ(√d_out)      ‖∇_{X}L_i‖₂ = Θ(√d_in)      ‖ΔY_i‖₂ = Θ(√d_out)     ∀i, ∀layer
```

含义：**每个元素的典型大小是 Θ(1)**，与我们要缩放的那个量（宽度 / 深度 / 稀疏度）无关。这样超参才能迁移。

注意 FLD 是**holistic** 的：必须同时控制前向、反向、更新三件事。论文 Sec.5 的核心论点是：既有工作各自只修了一部分（只修 init 的：Evci 2022、Liu 2018、Ramanujan 2020；只修梯度流的：Tessera 2021、Wang 2020；只修优化器的：Frantar 2023 对 Adafactor 的 RMS 修正），所以都不够。附录 C 的消融也证实：**只加 SµPar 的 init 或只加它的 LR 都不行**（Fig.13/14）。

## 2.3 前向（App. D.1）：为什么是 `σ_W² ∝ 1/(m_d m_ρ)`

掩码每列期望有 `d_in ρ` 个非零，于是

```
Y_ij = Σ_{k: M_kj = 1} X_ik W_kj
```

`E[W]=0` → `E[Y_ij] = 0`（均值自动受控）。方差：

```
Var(Y_ij) = Σ_{k:M=1} Var(X_ik W_kj)
          = d_in ρ · σ_W² · (Var(X) + E[X]²)                 (用 X⊥W, E[W]=0)
          = m_din · d_in,base · m_ρ · ρ_base · σ_W² · (Var(X)+E[X]²)
```

要求与 `m_din`、`m_ρ` 无关 →

```
σ_W² = σ_base² / (m_din · m_ρ)        ← 得到与 μP 一致的形式 + 密度因子
```

直觉：掩码把列上 `d_in` 个加数砍成 `d_in ρ` 个，方差按 `ρ` 线性掉；所以要给 `σ²` 乘 `1/ρ` 补回来（标准差乘 `1/√ρ`）。

**这与 Evci et al. 2022 的 sparsity-aware init 是同一个修正**（论文自己指出）。差别在于：他们只修了这个，没修更新。

## 2.4 反向（App. D.2）：同一个 σ_W² 同时管两向

掩码每行期望有 `d_out ρ` 个非零：

```
∇_X L_ij = Σ_{k: M_jk = 1} ∇_Y L_ik W_jk
Var(∇_X L_ij) = m_dout · d_out,base · m_ρ · ρ_base · σ_W² · Var(∇_Y L)
```

→ 同样要 `σ_W² = σ_base²/(m_dout m_ρ)`。

当 `m_din = m_dout`（常规做法），**一个初始化方差同时满足前向和反向**。这是推导里最省事的一步，也解释了为什么 μP 用单一 `σ_base` 就能管两头。

## 2.5 更新（App. D.3）：这里有真正的差别

```
ΔY_ij = η Σ_{k: M_kj = 1} X_ik ΔW_kj
E[ΔY_ij] → η · d_in ρ · E[X ΔW]              (LLN, d_in ρ → ∞)
```

**注意**：这里 `X` 与 `ΔW` 不独立（`ΔW` 由 `X` 算出），所以不能像前向那样直接拆开，只能靠大数定律取期望。

### SGD

```
ΔW = (1/d_in) Σ_b Xᵀ ∇_Y L          （已经带了 1/d_in 归一化）
⇒ E[ΔY] → η · m_ρ · ρ_base · E[X Σ_b X ∇_Y L]
```

与 `m_d` 无关（`d_in` 被 SGD 自己的 `1/d_in` 抵消了）→

```
η_sgd = η_base / m_ρ
```

### Adam

Adam 的更新 ≈ `η · 梯度 / √(梯度的二阶矩)`，即**逐元素的尺度被 RMS 归一化掉了**。于是 `ΔW` 的每个元素大小 ≈ `η`（与梯度尺度无关），但 `ΔY` 要对 `d_in ρ` 个元素求和：

```
E[ΔY_ij] → η · m_din · d_in,base · m_ρ · ρ_base · E[ X · (Σ_t γ_t Σ_b X∇Y) / √(Σ_t ω_t Σ_b (X∇Y)²) ]
```

求和项随 `d_in` 线性增长，必须用 `1/d_in` 抵掉 →

```
η_adam = η_base / (m_din · m_ρ)
```

**这是 μP 的核心洞察**（Yang et al. 2021）：Adam 下宽网络每层要用 `1/width` 的学习率，否则更新幅度随宽度爆炸。SµPar 只是把 `1/ρ` 再乘上去。

> 一句话记法：**SGD 的 `1/d_in` 藏在梯度里，Adam 的没有，要显式补。**

## 2.6 汇总表（论文 Table 1）

| 参数化 | 隐藏层 init 方差 | 隐藏层 LR (Adam) | 隐藏层 LR (SGD) |
|---|---|---|---|
| SP | `σ_base²` | `η_base` | `η_base` |
| μP | `σ_base² / m_d` | `η_base / m_d` | `η_base / m_d` |
| **SµPar** | `σ_base² / (m_d m_ρ)` | `η_base / (m_d m_ρ)` | `η_base / m_ρ` |

嵌入/解嵌入/LayerNorm/bias/attn logits：**三者相同**（按 μP 处理），因为论文只稀疏化投影层。

代码对应：`src/smupar/scaling.py`（`SparseScaling.init_std_factor` / `adam_lr_factor` / `sgd_lr_factor`）。

## 2.7 数值验证（本仓库实测）

`tests/test_smupar.py::test_supar_keeps_activation_variance_density_invariant`
：`d_in=1024, d_out=512, σ_base=0.02, X~N(0,I)`，密度取 `1, 0.5, 0.25, 0.0625`：

```
Var(Y)/Var(X) 在四个密度下的最大值/最小值 < 1.15   ✅
```

`test_sp_activations_shrink_with_sparsity`：SP 下 `Var(Y)` 严格随 ρ 递减 ✅

**端到端探针**（`scripts/run_supar_probe.py`，d_model=256、3 层残差 FFN、150 步，实测）：

| density | SP · mean\|pre-act\| (L0) | SµPar · mean\|pre-act\| (L0) |
|---|---|---|
| 1.000 | 2.554e-01 | 2.560e-01 |
| 0.500 | 1.798e-01 | 2.554e-01 |
| 0.250 | 1.266e-01 | 2.535e-01 |
| 0.125 | 8.889e-02 | 2.521e-01 |
| 0.0625 | 6.221e-02 | 2.507e-01 |

SP 的衰减精确符合 `√ρ`（`0.255×√(1/16) = 0.064` ✓），SµPar 在 16× 密度变化下几乎不动。**Finding 1 与 Finding 3 的机制被复现。**

图：`docs/smupar/supar_probe.png`，原始数据 `docs/smupar/supar_probe.csv`（生成命令见 README 第 5 条）。

⚠️ 注意：因为宽度固定等于 base（`m_d = 1`），本探针里 **μP ≡ SP**（实测两列几乎相同）。这不是 bug，而是 SµPar 的退化性质——也提醒我们：**只有同时改变宽度和密度时，μP 与 SµPar 才会分开**。

## 2.8 推导的薄弱环节

1. **`∇_Y L ⊥ W` 的假设**（App. D.2 脚注 3）：初始化时梯度与权重其实相关（从而不是严格独立的）。作者明说这是简化假设，留给未来工作。
2. **大数定律要求 `d_in ρ → ∞`**，即 WPN → ∞。在 99.2% 稀疏 + 宽度 2048 时 WPN ≈ 16，已经不是"很大"了。作者在 D.4 承认："If WPN is small, training dynamics may be affected."
3. **每层可独立设 ρ 和 η**：推导是逐层的，若采用层级不同稀疏度，必须逐层修正（许多剪枝方法正是这样）。
4. **只对线性投影层成立**：注意力 logits、LN、嵌入都没稀疏化，所以直接沿用 μP。
