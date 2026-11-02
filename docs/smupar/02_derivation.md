# 02 · 推导：三条 desiderata 与三个缩放因子

> 目标：把 App. D 的推导自己走一遍，确认每一步用到的假设，并用数值实验检验（本仓库 `tests/test_smupar.py`）。

## 2.1 记号

| 符号 | 含义 |
|---|---|
| $B,\ d_{\mathrm{in}},\ d_{\mathrm{out}}$ | batch、输入维度、输出维度 |
| $X \in \mathbb{R}^{B \times d_{\mathrm{in}}}$，$Y \in \mathbb{R}^{B \times d_{\mathrm{out}}}$ | 激活 |
| $W \in \mathbb{R}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}$，$M \in \{0,1\}^{d_{\mathrm{in}} \times d_{\mathrm{out}}}$ | 权重、静态随机掩码 |
| $\rho = 1 - s$ | 密度（1 = dense） |
| $\sigma_W^2$ | 权重初始化方差 |
| $m_d = d_{\mathrm{model}} / d_{\mathrm{model,base}}$，$m_\rho = \rho / \rho_{\mathrm{base}}$ | 宽度倍数、密度倍数 |
| $\Delta W$，$\Delta Y = X(\Delta W \odot M)$ | 权重更新、更新对输出的影响 |
| $\eta$ | 该层学习率 |

## 2.2 Feature Learning Desiderata（FLD）

$$
\lVert Y_i \rVert_2 = \Theta(\sqrt{d_{\mathrm{out}}}),
\qquad
\lVert \nabla_X L_i \rVert_2 = \Theta(\sqrt{d_{\mathrm{in}}}),
\qquad
\lVert \Delta Y_i \rVert_2 = \Theta(\sqrt{d_{\mathrm{out}}})
\quad \forall i,\ \forall \text{layer}
$$

含义：**每个元素的典型大小是 $\Theta(1)$**，与我们要缩放的那个量（宽度 / 深度 / 稀疏度）无关。这样超参才能迁移。

注意 FLD 是**holistic** 的：必须同时控制前向、反向、更新三件事。论文 Sec.5 的核心论点是：既有工作各自只修了一部分（只修 init 的：Evci 2022、Liu 2018、Ramanujan 2020；只修梯度流的：Tessera 2021、Wang 2020；只修优化器的：Frantar 2023 对 Adafactor 的 RMS 修正），所以都不够。附录 C 的消融也证实：**只加 SµPar 的 init 或只加它的 LR 都不行**（Fig.13/14）。

## 2.3 前向（App. D.1）：为什么是 $\sigma_W^2 \propto 1/(m_d m_\rho)$

掩码每列期望有 $d_{\mathrm{in}}\rho$ 个非零，于是

$$
Y_{ij} = \sum_{k:\, M_{kj}=1} X_{ik} W_{kj}
$$

$\mathbb{E}[W]=0$，所以 $\mathbb{E}[Y_{ij}]=0$（均值自动受控）。方差：

$$
\mathrm{Var}(Y_{ij})
= \sum_{k:\,M_{kj}=1} \mathrm{Var}(X_{ik}W_{kj})
= d_{\mathrm{in}}\rho \cdot \sigma_W^2 \cdot \big(\mathrm{Var}(X) + \mathbb{E}[X]^2\big)
\tag{用 $X \perp W,\ \mathbb{E}[W]=0$}
$$

$$
= m_{\mathrm{in}} \cdot d_{\mathrm{in,base}} \cdot m_\rho \cdot \rho_{\mathrm{base}} \cdot \sigma_W^2 \cdot \big(\mathrm{Var}(X) + \mathbb{E}[X]^2\big)
$$

要求与 $m_{\mathrm{in}}$、$m_\rho$ 无关，得到

$$
\sigma_W^2 = \frac{\sigma_{\mathrm{base}}^2}{m_{\mathrm{in}} \cdot m_\rho}
\qquad \text{（与 μP 一致的形式 + 密度因子）}
$$

直觉：掩码把列上 $d_{\mathrm{in}}$ 个加数砍成 $d_{\mathrm{in}}\rho$ 个，方差按 $\rho$ 线性掉；所以要给 $\sigma^2$ 乘 $1/\rho$ 补回来（标准差乘 $1/\sqrt{\rho}$）。

**这与 Evci et al. 2022 的 sparsity-aware init 是同一个修正**（论文自己指出）。差别在于：他们只修了这个，没修更新。

## 2.4 反向（App. D.2）：同一个 $\sigma_W^2$ 同时管两向

掩码每行期望有 $d_{\mathrm{out}}\rho$ 个非零：

$$
\nabla_X L_{ij} = \sum_{k:\, M_{jk}=1} \nabla_Y L_{ik} W_{jk},
\qquad
\mathrm{Var}(\nabla_X L_{ij}) = m_{\mathrm{out}} \cdot d_{\mathrm{out,base}} \cdot m_\rho \cdot \rho_{\mathrm{base}} \cdot \sigma_W^2 \cdot \mathrm{Var}(\nabla_Y L)
$$

同样要 $\sigma_W^2 = \sigma_{\mathrm{base}}^2 / (m_{\mathrm{out}} m_\rho)$。

当 $m_{\mathrm{in}} = m_{\mathrm{out}}$（常规做法），**一个初始化方差同时满足前向和反向**。这是推导里最省事的一步，也解释了为什么 μP 用单一 $\sigma_{\mathrm{base}}$ 就能管两头。

## 2.5 更新（App. D.3）：这里有真正的差别

$$
\Delta Y_{ij} = \eta \sum_{k:\, M_{kj}=1} X_{ik}\, \Delta W_{kj}
\;\;\Rightarrow\;\;
\mathbb{E}[\Delta Y_{ij}] \to \eta \cdot d_{\mathrm{in}}\rho \cdot \mathbb{E}[X\,\Delta W]
\quad (\text{LLN},\ d_{\mathrm{in}}\rho \to \infty)
$$

**注意**：这里 $X$ 与 $\Delta W$ 不独立（$\Delta W$ 由 $X$ 算出），所以不能像前向那样直接拆开，只能靠大数定律取期望。

### SGD

$$
\Delta W = \frac{1}{d_{\mathrm{in}}} \sum_b X^\top \nabla_Y L
\;\;\Rightarrow\;\;
\mathbb{E}[\Delta Y] \to \eta \cdot m_\rho \cdot \rho_{\mathrm{base}} \cdot \mathbb{E}\Big[X \sum_b X \nabla_Y L\Big]
$$

与 $m_d$ 无关（$d_{\mathrm{in}}$ 被 SGD 自己的 $1/d_{\mathrm{in}}$ 抵消了），所以

$$
\eta_{\mathrm{sgd}} = \frac{\eta_{\mathrm{base}}}{m_\rho}
$$

### Adam

Adam 的更新 $\approx \eta \cdot$ 梯度$/\sqrt{\text{二阶矩}}$，即**逐元素的尺度被 RMS 归一化掉了**。于是 $\Delta W$ 的每个元素大小 $\approx \eta$（与梯度尺度无关），但 $\Delta Y$ 要对 $d_{\mathrm{in}}\rho$ 个元素求和：

$$
\mathbb{E}[\Delta Y_{ij}] \to \eta \cdot m_{\mathrm{in}} \cdot d_{\mathrm{in,base}} \cdot m_\rho \cdot \rho_{\mathrm{base}} \cdot
\mathbb{E}\!\left[ X \cdot \frac{\sum_t \gamma_t \sum_b X \nabla_Y L}{\sqrt{\sum_t \omega_t \sum_b (X \nabla_Y L)^2}} \right]
$$

求和项随 $d_{\mathrm{in}}$ 线性增长，必须用 $1/d_{\mathrm{in}}$ 抵掉，所以

$$
\eta_{\mathrm{adam}} = \frac{\eta_{\mathrm{base}}}{m_{\mathrm{in}} \cdot m_\rho}
$$

**这是 μP 的核心洞察**（Yang et al. 2021）：Adam 下宽网络每层要用 $1/\text{width}$ 的学习率，否则更新幅度随宽度爆炸。SµPar 只是把 $1/\rho$ 再乘上去。

> 一句话记法：**SGD 的 $1/d_{\mathrm{in}}$ 藏在梯度里，Adam 的没有，要显式补。**

## 2.6 汇总表（论文 Table 1）

| 参数化 | 隐藏层 init 方差 | 隐藏层 LR (Adam) | 隐藏层 LR (SGD) |
|---|---|---|---|
| SP | $\sigma_{\mathrm{base}}^2$ | $\eta_{\mathrm{base}}$ | $\eta_{\mathrm{base}}$ |
| μP | $\sigma_{\mathrm{base}}^2 / m_d$ | $\eta_{\mathrm{base}} / m_d$ | $\eta_{\mathrm{base}}$ |
| **SµPar** | $\sigma_{\mathrm{base}}^2 / (m_d m_\rho)$ | $\eta_{\mathrm{base}} / (m_d m_\rho)$ | $\eta_{\mathrm{base}} / m_\rho$ |

> 注（SGD 列）：论文 App. D.3.1 明确写 SGD 的修正 $\eta = \eta_{\mathrm{base}} / m_\rho$，且 "equivalent to μP when $\rho=1$"——即 μP 下 SGD 学习率**不随宽度缩放**（$1/d_{\mathrm{in}}$ 已藏在 $\Delta W$ 的定义里，式 18）。本项目代码按此实现（`scaling.py::sgd_lr_factor`），与 Yang et al. 2021 原始 μP 论文的 SGD 约定需另行核对。

嵌入/解嵌入/LayerNorm/bias/attn logits：**三者相同**（按 μP 处理），因为论文只稀疏化投影层。

代码对应：`src/smupar/scaling.py`（`SparseScaling.init_std_factor` / `adam_lr_factor` / `sgd_lr_factor`）。

## 2.7 数值验证（本仓库实测）

`tests/test_smupar.py::test_supar_keeps_activation_variance_density_invariant`：
$d_{\mathrm{in}}=1024$，$d_{\mathrm{out}}=512$，$\sigma_{\mathrm{base}}=0.02$，$X \sim N(0, I)$，密度取 $1,\ 0.5,\ 0.25,\ 0.0625$：

$$
\frac{\max_\rho \mathrm{Var}(Y)/\mathrm{Var}(X)}{\min_\rho \mathrm{Var}(Y)/\mathrm{Var}(X)} < 1.15 \quad ✅
$$

`test_sp_activations_shrink_with_sparsity`：SP 下 $\mathrm{Var}(Y)$ 严格随 $\rho$ 递减 ✅

**端到端探针**（`scripts/run_supar_probe.py`，$d_{\mathrm{model}}=256$、3 层残差 FFN、150 步，实测）：

| density | SP · mean&#124;pre-act&#124; (L0) | SµPar · mean&#124;pre-act&#124; (L0) |
|---|---|---|
| 1.000 | 2.554e-01 | 2.560e-01 |
| 0.500 | 1.798e-01 | 2.554e-01 |
| 0.250 | 1.266e-01 | 2.535e-01 |
| 0.125 | 8.889e-02 | 2.521e-01 |
| 0.0625 | 6.221e-02 | 2.507e-01 |

SP 的衰减精确符合 $\sqrt{\rho}$（$0.255 \times \sqrt{1/16} = 0.064$ ✓），SµPar 在 16 倍密度变化下几乎不动。**Finding 1 与 Finding 3 的机制被复现。**

图：`docs/smupar/supar_probe.png`，原始数据 `docs/smupar/supar_probe.csv`（生成命令见 README 第 5 条）。

⚠️ 注意：因为宽度固定等于 base（$m_d = 1$），本探针里 **μP ≡ SP**（实测两列几乎相同）。这不是 bug，而是 SµPar 的退化性质——也提醒我们：**只有同时改变宽度和密度时，μP 与 SµPar 才会分开**。

## 2.8 推导的薄弱环节

1. **$\nabla_Y L \perp W$ 的假设**（App. D.2 脚注 3）：初始化时梯度与权重其实相关（从而不是严格独立的）。作者明说这是简化假设，留给未来工作。
2. **大数定律要求 $d_{\mathrm{in}}\rho \to \infty$**，即 WPN → ∞。在 99.2% 稀疏 + 宽度 2048 时 WPN ≈ 16，已经不是"很大"了。作者在 D.4 承认："If WPN is small, training dynamics may be affected."
3. **每层可独立设 $\rho$ 和 $\eta$**：推导是逐层的，若采用层级不同稀疏度，必须逐层修正（许多剪枝方法正是这样）。
4. **只对线性投影层成立**：注意力 logits、LN、嵌入都没稀疏化，所以直接沿用 μP。
