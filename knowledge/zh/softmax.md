## 背景

模型最后一层输出的是一组任意实数（logits），比如分类时每个类别一个分数。Softmax 把它们变成一个**概率分布**：每个值都在 $(0, 1)$ 之间，加起来等于 $1$，原来分数越大，概率越大。分类的输出层、注意力权重都靠它。

## 核心公式

$$
\operatorname{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j} e^{x_j}}
$$

**平移不变性**：给所有 $x_j$ 同时减去同一个常数 $c$，结果不变：

$$
\frac{e^{x_i - c}}{\sum_j e^{x_j - c}}
= \frac{e^{x_i}\, e^{-c}}{e^{-c} \sum_j e^{x_j}}
= \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

**数值稳定**：float32 下 $e^{89}$ 就已经溢出为 $\infty$，于是会出现 $\infty / \infty = \text{NaN}$。取 $c = \max_j x_j$，则所有指数 $x_j - c \le 0$，$e^{x_j - c} \in (0, 1]$，不会溢出；且最大的那一项恰好是 $e^0 = 1$，分母至少为 $1$，也不会除以零。

## 形状

输入任意形状，沿指定的 `dim` 做 softmax，其余维度相互独立。输出形状与输入相同，沿 `dim` 求和为 $1$。

## 用到的 PyTorch 操作

- `x.max(dim=d, keepdim=True)`：返回一个 `(values, indices)` 二元组，取 `.values`。
- `torch.exp`、`.sum(dim=d, keepdim=True)`。
- `keepdim=True` 让归约后的维度保留为长度 1，便于和原张量广播。

## 常见坑

- **不减最大值**：输入里有 $1000$ 这样的值就会得到 NaN，测试专门检查这种情况。
- **把 `x.max(...)` 直接拿来相减**：它返回的是二元组，要取 `.values`。
- **全局最大值**：`x.max()` 不带 `dim` 时减的是整个张量的最大值。数学上结果仍然正确，但某一行的值如果都远小于全局最大值，这一行会全部下溢成 $0$，得到 $0/0$。应当沿 `dim` 取最大值。
- **忘了 `keepdim`**：`dim` 不是最后一维时，广播会对错位置。

## 延伸

- **在网络里的位置**：分类的输出层、注意力权重、MoE 的路由打分，都要用到它。几乎总是沿最后一维、对每一行独立做。
- **log-softmax**：很多时候需要的是概率的对数。设 $m = \max_j x_j$，

$$
\log\operatorname{softmax}(x)_i = x_i - m - \log \sum_j e^{x_j - m}
$$

  后一项就是 log-sum-exp。先求 softmax 再取对数，小概率可能先下溢成 $0$，再得到 $-\infty$。所以交叉熵的实现直接用 log-softmax（见「损失函数」专题）。
- **在线 softmax**：不必先扫一遍求最大值、再扫一遍求和。一边读入 $x$，一边维护当前最大值 $m$ 和归一化和 $\ell$：

$$
m' = \max(m, x),\qquad \ell' = \ell\, e^{m - m'} + e^{x - m'}
$$

  新的最大值出现时，把旧的和按 $e^{m - m'}$ 缩小即可。这是 FlashAttention 分块计算注意力的基础（见「IO 感知 kernel」专题）。
- **温度**：$\operatorname{softmax}(x / T)$。$T \to 0$ 逼近 one-hot（取最大），$T \to \infty$ 趋向均匀分布。解码时调的 temperature 就是它（见「解码与采样」专题）。
- **梯度**：设 $p = \operatorname{softmax}(x)$，$g = \partial L / \partial p$，则

$$
\frac{\partial L}{\partial x_j} = p_j \Bigl(g_j - \sum_i p_i\, g_i\Bigr)
$$

  推导见「Softmax 反向传播」一题，属于「手写反向传播」专题。
- **代价与精度**：softmax 是访存受限的算子，要做一次取最大、一次求和。输入是 fp16 / bf16 时，指数和求和应当在 fp32 下进行，PyTorch 的 autocast 也是这样处理的。

## 面试追问

1. **为什么减去最大值不会改变结果？**
   softmax 对输入整体平移不变：分子分母同时乘以 $e^{-c}$，约掉了。所以可以任选 $c$，取最大值只是为了让指数不超过 $1$。
2. **softmax 对输入整体缩放也不变吗？**
   不变的是平移，缩放会改变结果。$x$ 乘以大于 $1$ 的系数，分布变尖；乘以小于 $1$ 的系数，分布变平。这正是温度，也是注意力里要除以 $\sqrt{d_k}$ 的原因。
3. **为什么交叉熵不直接用 softmax 再取对数？**
   一是小概率会下溢成 $0$，取对数得到 $-\infty$；二是 log-softmax 融合后，既更稳定，反向传播的梯度也更简单：$p - y$，其中 $y$ 是 one-hot 标签。
4. **softmax 的输出能取到 $0$ 或 $1$ 吗？**
   数学上不能，每一项都在开区间 $(0, 1)$ 里。浮点数里可以：差距很大时，小项下溢成 $0$，最大项舍入成 $1$。屏蔽用的 $-\infty$ 输入则恰好得到 $0$。
5. **softmax 的 Jacobian 是什么形状？秩是多少？**
   $J = \operatorname{diag}(p) - p\, p^{\top}$，是 $n \times n$ 的对称矩阵。因为平移不变，$J \mathbf{1} = 0$，所以秩至多为 $n - 1$。
