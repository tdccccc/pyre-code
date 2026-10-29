## 背景

全量微调一个大模型，要为每个参数保存梯度和 Adam 状态，显存开销是参数本身的好几倍；每个下游任务还要存一份完整的模型副本。

**LoRA**（Low-Rank Adaptation）的假设：微调时权重的**变化量** $\Delta W$ 是低秩的。于是冻结原始权重 $W_0$，只训练两个小矩阵，它们的乘积就是 $\Delta W$。

## 核心公式

$$
h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r}\, B A\, x
$$

- $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$：冻结，不训练
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$：先把输入降到 $r$ 维
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$：再升回输出维度
- $r \ll \min(d_{\text{in}}, d_{\text{out}})$：秩，常取 $4 \sim 64$

**参数量**：

$$
\underbrace{r\,(d_{\text{in}} + d_{\text{out}})}_{\text{LoRA}}
\quad\text{vs}\quad
\underbrace{d_{\text{in}}\, d_{\text{out}}}_{\text{全量}}
$$

$d_{\text{in}} = d_{\text{out}} = 4096$、$r = 8$ 时，LoRA 只有 $65536$ 个参数，是全量 $1677$ 万的 $0.4\%$。

**初始化**：$B = 0$，$A$ 随机。这样训练开始时 $\Delta W = BA = 0$，模型输出与原模型完全相同。两者不能都为零：如果 $A = 0$、$B = 0$，则

$$
\frac{\partial L}{\partial B} \propto (A x)^{\top} = 0,
\qquad
\frac{\partial L}{\partial A} \propto B^{\top}(\cdots) = 0
$$

梯度永远为零，什么也学不到。

**缩放 $\alpha / r$**：调整 $r$ 时，更新量的尺度保持大致稳定，不必重新调学习率。

**合并**：训练完成后可以算出 $W = W_0 + \frac{\alpha}{r} BA$ 替换原权重，推理时没有任何额外开销。

## 形状

| 量 | 形状 |
| --- | --- |
| `x` | `(*, in_features)` |
| `lora_A` | `(rank, in_features)` |
| `lora_B` | `(out_features, rank)` |
| $x A^{\top}$ | `(*, rank)` |
| $x A^{\top} B^{\top}$、输出 | `(*, out_features)` |

## 用到的 PyTorch 操作

- `nn.Linear`，`p.requires_grad_(False)` 冻结权重**和偏置**。
- `nn.Parameter(torch.randn(...) * 0.01)`、`nn.Parameter(torch.zeros(...))`。
- `x @ A.T @ B.T`：先降维再升维，不必显式算出 $d_{\text{out}} \times d_{\text{in}}$ 的 $BA$。

## 常见坑

- **只冻结了权重，没冻结偏置**：测试要求只有 LoRA 参数有梯度。
- **$A$ 也初始化为零**：梯度为零，无法训练。
- **矩阵乘的顺序或转置错误**：按上面的形状表核对。
- **漏了缩放** $\alpha / r$。
