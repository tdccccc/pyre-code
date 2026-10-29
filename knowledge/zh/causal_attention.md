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
