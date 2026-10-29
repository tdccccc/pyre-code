## 背景

Transformer 每一层里都有一个前馈网络（FFN / MLP）。最初的写法是「线性层 → 激活 → 线性层」。LLaMA、PaLM 等模型改用 **门控线性单元（GLU）** 的变体：把输入投影成两路，一路经过激活后当作「门」，逐元素地控制另一路能通过多少。SwiGLU 就是门控用 Swish 激活的版本。

## 核心公式

**Swish**（也叫 SiLU）：

$$
\operatorname{swish}(z) = z \cdot \sigma(z), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

**SwiGLU 前馈**：

$$
h = (x W_1) \odot \operatorname{swish}(x W_{\text{gate}}), \qquad
\text{out} = h\, W_2
$$

- $x$：输入，$W_1, W_{\text{gate}}$：两路「升维」投影，$W_2$：「降维」回原来的维度
- $\odot$：逐元素乘（Hadamard 积），不是矩阵乘

和普通 FFN $\;\text{out} = \operatorname{act}(x W_1)\, W_2\;$ 对比：多了一个门控矩阵 $W_{\text{gate}}$，门的值依赖输入本身，所以表达能力更强。

## 形状

| 量 | 形状 |
| --- | --- |
| $x$ | `(B, d_model)` |
| $x W_1$、$x W_{\text{gate}}$、$h$ | `(B, d_ff)` |
| $\text{out}$ | `(B, d_model)` |

矩阵乘 `(B, d_model) @ (d_model, d_ff)` 要求左边的列数等于右边的行数，结果是 `(B, d_ff)`。

## 用到的 PyTorch 操作

- `@` / `torch.matmul`：矩阵乘。
- `torch.sigmoid(z)`：逐元素 $\sigma(z)$。
- `*`：逐元素乘，即 $\odot$。

## 常见坑

- **把 $\odot$ 写成矩阵乘**：两路都是 `(B, d_ff)`，它们之间是逐元素乘 `*`，不是 `@`。
- **门控放错一路**：Swish 作用在 $x W_{\text{gate}}$ 上，$x W_1$ 不过激活。两者对调结果就不对了。
- **只乘了 sigmoid**：Swish 是 $z \cdot \sigma(z)$，不是 $\sigma(z)$ 本身。
- 这里的权重已经是 `(in, out)` 布局，直接 `x @ W` 即可，不需要转置（和 `nn.Linear` 的 `(out, in)` 布局不同）。
