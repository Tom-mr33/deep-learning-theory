# deep-learning-theory

深度学习理论精读 + 复现工程。当前聚焦两篇关于**训练动力学**的论文：

| # | 论文 | 出处 | 一句话 | 目录 |
|---|------|------|--------|------|
| 1 | **Looped Transformers are Better at Learning Learning Algorithms** | ICLR 2024 · arXiv:2311.12424 | 单层 Transformer 加"输入注入"+循环复用，0.79M 参数追平 9.48M 的 12 层模型，在稀疏线性/决策树上反超 | [`docs/looped_transformer`](docs/looped_transformer) · [`src/looped`](src/looped) |
| 2 | **Sparse maximal update parameterization (SµPar)** | arXiv:2405.15743 (Cerebras) | 把 μP 从"随宽度缩放"推广到"随宽度 × 稀疏度缩放"，让 dense 上调好的超参直接迁移到 99.2% 稀疏模型 | [`docs/smupar`](docs/smupar) · [`src/smupar`](src/smupar) |

两篇的公共母题：**信号传播 / 有效深度 / 参数复用 / 超参迁移**。第一篇问"能否用更少的参数实现迭代算法的深度"，第二篇问"稀疏化破坏了哪些尺度不变性，怎么系统地修回来"。

## 目录结构

```
docs/                       精读笔记（逐节拆解 + 逐图笔记 + 批判 + 复现计划）
  looped_transformer/       01_overview 02_method 03_experiments 04_critique 05_reproduction_plan
  smupar/                   01_overview 02_derivation 03_experiments 04_critique
src/looped/                 复现代码：data(任务生成) model(循环架构) solvers(OLS基线) train(窗口损失+调度)
src/smupar/                 SµPar：scaling(缩放律) layers(稀疏线性层) mask(随机掩码) probe(激活尺度探针)
src/common/                 配置/日志/随机种子等公共工具
configs/                    实验配置（debug / smoke / paper 设置）
scripts/                    CLI 入口
tests/                      单元测试（18 项，含 SµPar 方差不变性与截断 BPTT 的数值验证）
papers/                     PDF（不入库，见 .gitignore）
results/                    训练日志 jsonl + 汇总 json + 探针图（部分不入库）
```

## 快速开始

```bash
pip install -r requirements.txt

# 1) 单元测试（约 15s）：验证数据布局、OLS 正确性、参数共享、SµPar 缩放律
python -m pytest tests -q

# 2) 管道冒烟（约 1-2 分钟 CPU/GPU）：只验证能跑通，不产出论文数字
python scripts/train_looped.py --config configs/looped_linear_debug.yaml

# 3) 小规模可学习 demo（若干分钟）
python scripts/train_looped.py --config configs/looped_linear_smoke.yaml

# 4) 论文设置（需要 GPU，数小时）
python scripts/train_looped.py --config configs/looped_linear.yaml --device cuda
python scripts/train_looped.py --config configs/standard_linear.yaml --device cuda

# 5) SµPar 激活尺度探针（复现 Fig.5 的机制，非 LLM 规模）
python scripts/run_supar_probe.py --steps 200 --d-model 256
```

## 复现状态

| 项目 | 状态 |
|------|------|
| 数据管线（linear / sparse linear / decision tree / 2-layer NN）、前缀一次前向算 k+1 个训练信号 | ✅ 已验证（单批次过拟合 err≈5e-3） |
| 循环架构 `Y_{t+1}=M(Y_t+P)`、参数共享、有效深度、逐迭代输出 | ✅ 已验证（参数量 <15% 于 12 层） |
| 窗口截断损失 `[b-T, b]` + b 调度 | ✅ 已实现（`detach_before` 真正截断反传图，测试验证窗口外迭代无梯度） |
| **端到端小规模复现**（`configs/looped_linear_smoke.yaml`，30 s）：算法被学到 + 超过 b 后退化 | ✅ 实测 b=3 → 0.067；b=6 → 0.12；b=12 → 0.26；OLS 1.6e-12 |
| 论文规模数字（Fig.2 / Fig.10 / Table 2） | ⏳ 未跑（需 GPU，配置已就绪；d=20/k=40 需要 1e4–1e5 步） |
| SµPar 缩放律（init / Adam LR / SGD LR） | ✅ 数值验证：density 从 1 → 1/16，激活方差漂移 <15% |
| SµPar 激活尺度探针（Fig.5 机制） | ✅ 实测 SP 按 `√ρ` 衰减（0.255→0.062），SµPar 保持 0.25 不变，见 `docs/smupar/supar_probe.png` |
| SµPar 的 LLM 结论（11.9% loss 改进、HP 迁移） | ⏳ 不做（需要 Cerebras 级算力），只做机制层验证 + 读图笔记 |

诚实说明：本仓库的价值在**可运行的复现骨架 + 可验证的机制检验**，不是刷论文数字。所有未验证的结论在笔记里都标了 "未复现"。

小规模 demo 的一个副产品：`b=3` 训练、推理跑到 12 次循环时误差从 0.067 涨到 0.26 —— 正好复现论文 Fig.4 的"**b 太小就没有稳定不动点**"。

## 环境

- Python 3.14 + PyTorch 2.11（`torch.nn.functional.scaled_dot_product_attention`）
- CPU 可跑通全部管线；论文级实验建议单卡 GPU（8GB 足够跑 d=20/k=40 的设置）

## Roadmap

1. `scripts/run_looped_sweep.py`：b × T 网格扫描，自动生成 Fig.4 风格曲线
2. 权重绑定（无输入注入）对照实验，验证 Fig.3 的发散
3. 层间 probing（复现 Fig.8：X^T y 与 w_OLS 的可解码性）
4. OOD 测试套件：偏斜协方差 / d/2 子空间 / x 缩放（复现 Fig.6、Fig.15、Fig.16）
5. SµPar 扩展到 2:4 结构化稀疏与 Iso-WPN 缩放曲线的验证
6. 把两个母题接起来：**循环模型的有效深度** × **稀疏化的信号尺度**（见 `docs/00_reading_plan.md` 的选题 3）

## 参考

- Yang, Lee, Nowak, Papailiopoulos. *Looped Transformers are Better at Learning Learning Algorithms.* ICLR 2024.
- Dey, Bergsma, Hestness. *Sparse maximal update parameterization.* Cerebras Systems, 2024.
- 上游代码：<https://github.com/Leiay/looped_transformer>、<https://github.com/EleutherAI/nanoGPT-mup/tree/supar>
- 前置：Garg et al. 2022（in-context 函数类）、Yang & Hu 2020 / Yang et al. 2021（μP）
