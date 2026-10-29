## 背景

fp32 权重每个数占 4 字节。**量化**把它们用 int8（1 字节）存储，模型体积缩小为 $1/4$。推理时大模型的瓶颈常常是从显存读权重，读的字节少了，速度也就上去了。代价是一点精度损失。

## 核心公式

**对称量化**：用一个缩放因子 $s$ 把实数映射到整数网格：

$$
q = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right),\ -128,\ 127\right),
\qquad
\hat{w} = q \cdot s
$$

$\hat{w}$ 是反量化后的近似值。取 $s = \max|w| / 127$，让绝对值最大的元素正好映射到 $\pm 127$，整数范围被充分利用。

**误差**：只要没有被截断，四舍五入的误差不超过半个网格：

$$
|w - \hat{w}| \le \frac{s}{2}
$$

**逐通道（per-channel）**：权重矩阵 $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ 每一行（每个输出通道）用自己的缩放因子：

$$
s_j = \frac{\max_i |W_{j,i}|}{127}
$$

如果整个矩阵共用一个 $s$，只要某一行有一个很大的离群值，$s$ 就会很大，其余所有行的分辨率都被拉低。逐通道让每一行按自己的范围量化。

**前向**：$y = x\, \hat{W}^{\top} + b$，$\hat{W}_{j,:} = q_{j,:} \cdot s_j$。

## 形状

| 量 | 形状 |
| --- | --- |
| `weight`、`weight_int8` | `(out_features, in_features)` |
| `scale` | `(out_features, 1)`，保留维度以便按行广播 |
| `x` | `(*, in_features)` |
| 输出 | `(*, out_features)` |

## 用到的 PyTorch 操作

- `w.abs().amax(dim=1, keepdim=True)`：每行绝对值的最大值。
- `torch.round`、`t.clamp(-128, 127)`、`t.to(torch.int8)`。
- `self.register_buffer(name, tensor)`：注册**缓冲区**。它会随 `state_dict` 保存、随 `.to(device)` 移动，但不是参数，优化器不会更新它。
- `t.float()`：int8 不能直接和浮点张量做矩阵乘，要先转回浮点。

## 常见坑

- **用全局缩放因子**：误差会被离群值放大，测试检查的是逐通道。
- **缩放沿错误的维度**：`dim=1` 得到每个输出通道（每行）一个缩放因子；`dim=0` 就成了按输入通道。
- **全零行**：$\max|w| = 0$ 时 $s = 0$，除以零得到 NaN。可以给分母加一个极小值。
- **把量化权重注册成 `nn.Parameter`**：题目要求是 buffer；而且整数张量不能求梯度。
- **先 `to(int8)` 再 `round`**：类型转换会直接截断小数部分，而不是四舍五入。
