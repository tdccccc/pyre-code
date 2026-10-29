## 背景

GPT 这样的**自回归**语言模型逐个预测下一个 token，序列的概率按链式法则分解：

$$
p(x_1, \dots, x_S) = \prod_{t=1}^{S} p(x_t \mid x_1, \dots, x_{t-1})
$$

预测第 $t$ 个 token 时只能用前面的 token。训练时整个序列一次性送进模型并行计算，为了不「偷看答案」，第 $i$ 个位置的注意力必须只覆盖位置 $j \le i$。这就是**因果掩码**。

## 核心公式

$$
\operatorname{CausalAttn}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}} + M\right) V,
\qquad
M_{ij} =
\begin{cases}
0, & j \le i \\
-\infty, & j > i
\end{cases}
$$

$e^{-\infty} = 0$，被屏蔽位置的权重恰好为 $0$；softmax 只在允许的位置上归一化，每行权重之和仍为 $1$。

$S = 4$ 时，要屏蔽的位置是严格上三角：

$$
\begin{pmatrix}
\cdot & \times & \times & \times \\
\cdot & \cdot & \times & \times \\
\cdot & \cdot & \cdot & \times \\
\cdot & \cdot & \cdot & \cdot
\end{pmatrix}
\qquad (\times \text{ 为屏蔽})
$$

第 $0$ 行只有对角线一个位置可见，所以位置 $0$ 的输出就等于 $v_0$。

## 形状

| 量 | 形状 |
| --- | --- |
| `Q`、`K`、`V` | `(B, S, D)` |
| 分数 | `(B, S, S)` |
| 掩码 | `(S, S)`，`unsqueeze(0)` 后广播到 `(B, S, S)` |
| 输出 | `(B, S, D)` |

## 用到的 PyTorch 操作

- `torch.ones(S, S, dtype=torch.bool)`、`torch.triu(t, diagonal=1)`：严格上三角（不含对角线）。
- `scores.masked_fill(mask, float('-inf'))`：`mask` 为 `True` 的位置填 $-\infty$。
- `torch.bmm`、`torch.softmax`、`K.transpose(1, 2)`。

## 常见坑

- **用 0 代替 $-\infty$**：分数为 $0$ 时 $e^0 = 1$，被屏蔽的位置仍然有权重。
- **softmax 之后再把权重清零**：每行之和就不再是 $1$ 了，除非重新归一化。
- **`diagonal` 参数错**：`diagonal=0` 会把对角线也屏蔽，第 $0$ 行全是 $-\infty$，softmax 得到 NaN。
- **屏蔽了下三角**：方向反了，变成只能看未来。
- 掩码要和分数在同一个 device 上。

## 延伸

- **训练并行，推理逐步**：训练时一次把整个序列送入，因果掩码让每个位置都只看自己之前的内容，所以 $S$ 个位置同时得到 $S$ 个「预测下一个 token」的训练信号。推理时一次生成一个 token。因为每一层在位置 $j$ 的 K、V 只依赖 $j$ 及之前的 token，之前位置的 K、V 不会再变，可以缓存下来，新 token 只需要算最后一行，这就是 KV cache（见「推理服务」专题）。
- **有效计算只有一半**：下三角之外的分数都会被丢掉。朴素实现仍把整个 $S \times S$ 算完，高效的 kernel 则按块判断，整块在上三角里就直接跳过，计算量约省一半。
- **$S_q \ne S_k$ 时的对齐**：带缓存地预填充一段新 token、或解码时，查询比键短。此时合理的规则是**右下角对齐**：第 $i$ 个查询可以看到 $j \le i + (S_k - S_q)$。本题 $S_q = S_k$，`triu(diagonal=1)` 没问题，但换成别的形状就不再适用。不同实现的约定不同：PyTorch 的 `scaled_dot_product_attention(..., is_causal=True)` 按左上角对齐，FlashAttention 从 2.1 版起按右下角对齐，使用时需要留意。
- **双向与因果**：BERT 一类编码器是双向的，只需要 padding 掩码；GPT 一类解码器是因果的。Prefix-LM 介于两者之间，前缀部分内部双向，后面的部分因果。
- **$-\infty$ 与大负数**：有些实现用 `finfo.min` 代替 $-\infty$，避免整行被屏蔽时出现 NaN。代价是这种行得到的是均匀权重，而不是报错，错误更容易被掩盖。
- **调用**：`F.scaled_dot_product_attention(q, k, v, is_causal=True)` 不需要显式构造掩码。

## 面试追问

1. **为什么掩码要在 softmax 之前加，而不是之后清零？**
   之后清零，每行的和就不再是 $1$，而且被屏蔽位置的分数已经参与了归一化。softmax 之前填 $-\infty$，被屏蔽位置的权重恰好是 $0$，剩下的位置自动重新归一化。
2. **训练时有因果掩码，推理时为什么可以用 KV cache？**
   因果性保证了位置 $j$ 的 K、V 只由前 $j$ 个 token 决定，之后生成新 token 不会改变它们，所以算过一次就能一直复用。如果是双向注意力，每新增一个 token，所有位置的表示都要重算。
3. **位置 $0$ 的输出是什么？**
   位置 $0$ 只能看见自己，softmax 在一个元素上的结果是 $1$，输出就等于 $v_0$。
4. **因果注意力的计算量比双向少一半吗？**
   理论上有效的分数只有一半。朴素实现不会少，因为仍然算了整个矩阵再屏蔽；只有按块跳过上三角的 kernel 才真正省下。
5. **编码器-解码器模型里，哪些注意力要因果掩码？**
   解码器的自注意力要，防止偷看后面的译文；解码器对编码器的交叉注意力不要，原文是完整给定的；编码器的自注意力也不要（见「多头交叉注意力」一题）。
