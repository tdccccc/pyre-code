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

## 延伸

- **来源**：门控线性单元 GLU 最早由 Dauphin 等人在语言模型的卷积网络里提出，形式是 $(xW) \odot \sigma(xV)$。Shazeer（2020）把 sigmoid 换成其他激活函数，得到 ReGLU、GEGLU、SwiGLU 等一族变体，并在 Transformer 的 FFN 上做了比较，门控版本在参数量相当时效果更好。
- **为什么有效**：门控让每个隐藏单元的输出是两个线性函数的**乘积**，比单路激活多了一层「输入相关的缩放」，表达能力更强。这个解释偏直觉，实验结果比理论解释走得更远。
- **隐藏维缩成 $\tfrac{8}{3}d$**：门控 FFN 有三个矩阵而不是两个，为了与「隐藏维 $4d$ 的普通 FFN」保持相同的参数量，隐藏维取 $\tfrac{8}{3}d$，推导见「基础层与激活」专题。
- **计算量**：每个 token 前向有三次矩阵乘，约 $6\, d\, d_{\text{ff}}$ FLOPs。取 $d_{\text{ff}} = \tfrac{8}{3} d$ 时是 $16\, d^2$，和普通 FFN（隐藏维 $4d$）一致。
- **实现上的优化**：
  - 把 $W_1$ 和 $W_{\text{gate}}$ 沿输出维拼在一起，一次矩阵乘算出两路，再切开，GPU 利用率更高。
  - 「SiLU 再乘」这一步是纯逐元素运算，生产代码里会写成一个融合 kernel，避免把中间结果写回显存。
- **张量并行**：两个升维矩阵按列切分，降维矩阵按行切分，两次矩阵乘之间的逐元素运算不需要通信，只在最后做一次 all-reduce（见「分布式训练」专题）。
- **MoE 中的专家**：Mixtral、DeepSeek 等混合专家模型里，每个专家通常就是这样一个 SwiGLU FFN（见「模型架构」专题）。
- **命名对应**：本题的函数形式与「SwiGLU MLP」一题的模块形式是同一个计算，对应关系是 $W_1 \leftrightarrow$ `up_proj`、$W_{\text{gate}} \leftrightarrow$ `gate_proj`、$W_2 \leftrightarrow$ `down_proj`。

## 面试追问

1. **SwiGLU 相比普通 GELU FFN 改了什么？为什么更好？**
   多了一个投影，两路相乘，激活只作用在门控那一路。门控引入了输入相关的乘性交互。Shazeer 的实验里，在参数量相当时效果更好，但缺乏严格的理论解释。
2. **为什么 SwiGLU 的 $d_{\text{ff}}$ 常取 $\tfrac{8}{3} d$？**
   三个矩阵共 $3\, d\, d_{\text{ff}}$ 个参数，要和隐藏维 $4d$ 的两矩阵 FFN（$8d^2$）持平，所以 $d_{\text{ff}} = \tfrac{8}{3}d$。实际还会取整到 $256$ 之类的倍数。
3. **Swish、SiLU 有什么区别？**
   Swish 是 $z\,\sigma(\beta z)$，有一个参数 $\beta$；SiLU 是 $\beta = 1$ 的特例。LLM 里用的都是 $\beta = 1$，所以两个名字常常混用。
4. **SwiGLU 的 FLOPs 和参数量是多少？**
   参数量 $3\, d\, d_{\text{ff}}$（不含偏置），每个 token 前向约 $6\, d\, d_{\text{ff}}$ FLOPs。
5. **SwiGLU 的 FFN 怎么做张量并行？**
   gate 和 up 按列切（每张卡算一部分隐藏单元），down 按行切。逐元素的门控乘法在本地完成，只需要在 down 之后做一次 all-reduce。
