## 背景

**DiT**（Diffusion Transformer，Sora、SD3 的基础架构）用 Transformer 代替 U-Net 做去噪网络。每个块都需要知道「现在是第几个时间步」「要生成哪一类」，即**条件信息**。

**adaLN**（自适应层归一化）的做法：LayerNorm 本来有可学习的缩放 $\gamma$ 和平移 $\beta$，现在不再是固定参数，而是由条件嵌入**回归**出来，每个样本各不相同。**adaLN-Zero** 再加一个门控 $\alpha$ 控制残差分支的输出，并把它初始化为 $0$。

## 核心公式

**回归调制参数**：条件嵌入 $c$（时间步嵌入 + 类别嵌入）经过一个线性层，得到 6 组参数：

$$
[\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2] = c\, W + b
$$

**调制**（本题实现第一组）：

$$
\operatorname{LN}(x) = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}},
\qquad
\text{out} = \alpha_1 \odot \bigl(\gamma_1 \odot \operatorname{LN}(x) + \beta_1\bigr)
$$

这里的 LayerNorm 没有自己的可学习参数。

**在完整 DiT 块中**，两组参数分别用于注意力和 MLP 两个子层：

$$
\begin{aligned}
x &\leftarrow x + \alpha_1 \odot \operatorname{Attn}\bigl(\gamma_1 \odot \operatorname{LN}(x) + \beta_1\bigr) \\
x &\leftarrow x + \alpha_2 \odot \operatorname{MLP}\bigl(\gamma_2 \odot \operatorname{LN}(x) + \beta_2\bigr)
\end{aligned}
$$

**为什么零初始化**：$W$、$b$ 初始化为 $0$，则 $\alpha = 0$，每个残差分支的输出都是 $0$：

$$
x + \underbrace{0 \odot f(\cdot)}_{= 0} = x
$$

训练开始时每个块都是**恒等映射**，信号和梯度可以无损地穿过很深的网络，训练更稳定。网络在训练中再逐渐学会「打开」每个分支。

## 形状

| 量 | 形状 |
| --- | --- |
| `x` | `(B, N, D)` |
| `cond` | `(B, D)` |
| `W_ada`、`b_ada` | `(D, 6D)`、`(6D,)` |
| 回归出的参数 | `(B, 6D)`，切成 6 块，每块 `(B, D)` |
| 每块 `unsqueeze(1)` 后 | `(B, 1, D)`，广播到所有 $N$ 个 token |

## 用到的 PyTorch 操作

- `t.chunk(6, dim=-1)`：沿最后一维均分成 6 块。
- `t.unsqueeze(1)`：每个样本的调制参数作用于它的所有 token。
- `x.mean(dim=-1, keepdim=True)`、`x.var(dim=-1, keepdim=True, unbiased=False)`。

## 常见坑

- **没有 `unsqueeze(1)`**：`(B, D)` 和 `(B, N, D)` 直接相乘会按错误的维度广播，或者报错。
- **6 块的顺序**：$\gamma_1, \beta_1, \alpha_1, \gamma_2, \beta_2, \alpha_2$。
- **LayerNorm 的细节**：有偏方差、$\epsilon = 10^{-6}$ 加在根号内。
- 原始 DiT 论文用 $(1 + \gamma)$ 作缩放，让零初始化时缩放为 $1$；本题按题面直接用 $\gamma$。
