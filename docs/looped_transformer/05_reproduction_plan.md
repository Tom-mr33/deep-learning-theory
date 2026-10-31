# 05 · 复现计划与实现对照

## 5.1 代码 ↔ 论文的对应

| 论文元素 | 位置 | 说明 |
|---|---|---|
| `Y_{t+1} = M_θ(Y_t + P)`, `Y_0 = 0` | `src/looped/model.py::InContextTransformer.forward` | `looped=True` 时循环，`looped=False` 时单次前向 |
| 循环内的 `M_θ`（共享权重） | `TransformerBody` | 用 `nn.ModuleList` 只建 `n_layer` 个块；循环不复制参数 |
| prompt 编码 `P` | `emb_x / emb_y / pos_emb` | `x` 与标量标签分别线性嵌入后相加 |
| 读出头 | `ln_f + readout` | 每个位置都输出一个标量预测 |
| Eq.1 的窗口平均 | `train.py::windowed_loss` | 对 `t ∈ [b0, b)` 与所有被 mask 的位置取平均 |
| `b0 = max(b−T, 0)` | `train.py::train`（`b0 = max(b - T, 0)`） | |
| `b` 调度 | `train.py::current_b` | 线性 ramp，`schedule_b`/`b_start`/`b_warmup_steps` |
| `k` curriculum | `train.py`（`curriculum_k`） | 每步 `k ~ U{1..k_max}` |
| 四类函数类 | `src/looped/data.py` | linear / sparse linear / decision tree / 2-layer ReLU |
| OLS 基线 | `src/looped/solvers.py::ols_predict` | 用于 Fig.2 右的对照线 |
| 归一化误差 | `solvers.py::normalized_squared_error` | `MSE / Var(y)` |

## 5.2 与论文实现的已知偏差（写论文时必须声明）

1. **一次前向得到 k+1 个训练信号**。论文描述为"对每个前缀 `P^i` 分别算"，我们利用**因果掩码**在偶数位置同时读取全部前缀的预测（`data.py` 的布局注释）。数学等价，算力省 `k` 倍。⚠️ 但两者对注意力的归一化细节不同（我们的注意力在更长序列上做 softmax），严格说不是逐比特等价——若要完全对齐，需开关 `prefix_mode="per_prefix"`（未实现）。
2. **`d` 的 curriculum 未实现**（固定 `d`），只实现了 `k` 的 curriculum。理由：`d` 变化需要输入 padding + mask，收益对本项目不重要。
3. **GPT-2 块带 bias**（`bias=True`），论文未说明；dropout 固定 0（论文也未用）。
4. **误差归一化用 `MSE / Var(y_true)`**，论文沿用 Garg 的归一化，细节略有差异；绝对值可能有常数差，趋势不受影响。
5. **bootstrap 置信区间未实现**（论文用 1000 次重采样报 90% CI）。当前只报点估计。
6. 论文的 Adam `lr=1e-4` 无 weight decay / 无梯度裁剪；我们默认 `grad_clip=1e9`（等价于不裁剪），`weight_decay=0`。

## 5.3 实验矩阵（按优先级）

| 阶段 | 实验 | 配置 | 目的 | 成本估计 |
|---|---|---|---|---|
| P0 | 单元测试 | `pytest tests` | 数据布局、OLS、参数共享 | 15 s |
| P0 | 管道冒烟 | `configs/looped_linear_debug.yaml` | 能跑通 | 1–2 min |
| P1 | **核心主张**：looped vs 12 层 | `looped_linear.yaml` + `standard_linear.yaml` | 复现 Fig.2 右 | 单卡数小时 |
| P1 | 输入注入消融 | 同上，`model.looped=true` 但改成权重绑定（需加开关） | 复现 Fig.3 | 同上 |
| P2 | (b,T) 网格 | 5×3 组合 | 复现 Fig.4，找饱和点 | 15× P1 |
| P2 | 等 FLOPs 对照 | 自定义（见 04-A） | **论文没做，我们补** | ≈ P1 |
| P3 | probing | 需新增 `probe.py`（论文 App.F） | 复现 Fig.8 | 中等 |
| P3 | 样本效率 | 预生成 80k prompt | 复现 Fig.5 | 中等 |
| P4 | 稀疏/树/MLP | `looped_sparse_linear.yaml` 等 | 复现 Fig.10 | 大 |

## 5.4 关键决策点

- **要不要追数字？** 我的建议：不追。d=20/k=40 的设置下，达到 `10⁻³` 级误差通常需要 2×10⁴–10⁵ 步，纯 CPU 不可行；用单卡 GPU 也要数小时到一天。先把 **趋势正确**（looped ≤ 12 层、b 过小会发散）作为通过标准。
- **通过标准（建议）**：
  1. looped (b=20) 在 `k=10..40` 上的曲线与 12 层差值 < 20%（相对）
  2. looped 在 `b` 之后的迭代误差不上升（固定点稳定）
  3. `b=8` 的模型在 `t=40` 时明显发散（复现"b 太小"的现象）
- **规模缩小的原则**：先缩 `d`（20→5/10）和 `D`（256→64/128），保持 `k_max` 与 `b/T` 的比例关系；不要在缩小模型的同时改 `T/b` 的比例，否则会误判"固定点不稳定"。

## 5.5 尚未实现的模块（Roadmap）

- [ ] `scripts/run_looped_sweep.py`：b×T 网格自动扫描 + 画曲线（直接产出 Fig.4）
- [ ] `model.py` 加 `injection: "add" | "concat" | "none"`，支持权重绑定消融
- [ ] `src/looped/probe.py`：论文 App.F 的 probing（softmax 位置权重 + 2 层 MLP，目标 `Xᵀy` 与 `w_OLS`）
- [ ] `src/looped/ood.py`：偏斜协方差 / d/2 子空间 / 正交查询 / 象限 / 缩放
- [ ] `scripts/run_flops_compare.py`：等 FLOPs 前沿（我们自己的实验，见 04-A）
- [ ] bootstrap 90% CI
