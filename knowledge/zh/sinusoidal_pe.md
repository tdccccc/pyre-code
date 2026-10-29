## 背景

注意力对输入顺序是「无感」的：把序列打乱，每个 token 得到的注意力结果只是跟着换了位置，数值完全一样。模型要知道词的顺序，就必须把**位置信息**显式加进去。

原始 Transformer 的做法是给每个位置 $\text{pos}$ 生成一个 $d_{\text{model}}$ 维的固定向量，直接加到词嵌入上：

$$
x_{\text{pos}} = \text{Embed}(\text{token}_{\text{pos}}) + \operatorname{PE}(\text{pos})
$$

## 核心公式

维度两两成对，第 $i$ 对（$i = 0, 1, \dots, \tfrac{d}{2} - 1$）使用频率

$$
\omega_i = \frac{1}{10000^{2i/d}}
$$

偶数列放正弦、奇数列放余弦：

$$
\operatorname{PE}(\text{pos}, 2i) = \sin(\omega_i \cdot \text{pos}),
\qquad
\operatorname{PE}(\text{pos}, 2i+1) = \cos(\omega_i \cdot \text{pos})
$$

**多尺度**：$\omega_0 = 1$（变化最快，周期 $2\pi$），$\omega_i$ 随 $i$ 几何递减，最后一对的周期接近 $2\pi \cdot 10000$。就像时钟的秒针、分针、时针，不同维度在不同尺度上编码位置。

**相对位置是线性变换**：由三角恒等式，

$$
\begin{pmatrix} \sin(\omega(\text{pos}+k)) \\ \cos(\omega(\text{pos}+k)) \end{pmatrix}
=
\begin{pmatrix} \cos \omega k & \sin \omega k \\ -\sin \omega k & \cos \omega k \end{pmatrix}
\begin{pmatrix} \sin(\omega\, \text{pos}) \\ \cos(\omega\, \text{pos}) \end{pmatrix}
$$

偏移 $k$ 对应一个只依赖 $k$ 的旋转矩阵，模型因此容易学到相对位置关系。

## 形状

| 量 | 形状 |
| --- | --- |
| 位置 $\text{pos}$ | `(seq_len, 1)` |
| 频率 $\omega$ | `(d_model/2,)` |
| 角度 $\text{pos} \cdot \omega$（广播外积） | `(seq_len, d_model/2)` |
| 输出 PE | `(seq_len, d_model)` |

## 用到的 PyTorch 操作

- `torch.arange(n)`、`torch.arange(0, d, 2)`：生成 $0, 1, \dots$ 或偶数序列。
- `t.unsqueeze(1)` / `t[:, None]`：加一个维度，配合广播实现外积 $\text{pos}_p \cdot \omega_i$。
- `torch.sin`、`torch.cos`。
- 步长切片赋值：`pe[:, 0::2] = ...` 写偶数列，`pe[:, 1::2] = ...` 写奇数列。

## 常见坑

- **指数用错**：分母指数是 $2i/d$，其中 $2i$ 是偶数列的列号（$0, 2, 4, \dots$）。如果用 $i/d$ 且 $i$ 取 $0 \dots d/2-1$，频率就不对了。
- **整数运算**：`torch.arange` 默认是整数，整数张量的幂和除法可能截断或报错，记得转 `float`。
- **sin / cos 摆放方式**：本题是交错排列（偶数列 sin、奇数列 cos），不是前一半 sin、后一半 cos。
- 自检：$\text{pos} = 0$ 时所有 sin 列为 $0$、cos 列为 $1$；所有值都在 $[-1, 1]$。
