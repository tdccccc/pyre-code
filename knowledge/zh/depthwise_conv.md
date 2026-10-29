## 背景

标准卷积同时做两件事：在空间上聚合邻域信息，以及在通道之间混合信息。**深度可分离卷积**（MobileNet、Xception 使用）把它们拆成两步：

1. **深度卷积**（depthwise）：每个输入通道用自己的一个 $k \times k$ 卷积核，只做空间聚合，通道之间不交流
2. **逐点卷积**（pointwise）：$1 \times 1$ 卷积，只在每个像素位置上混合通道

参数量和计算量大幅下降，非常适合手机等算力有限的设备。

## 核心公式

**深度卷积**（每个通道 $c$ 独立）：

$$
z_{b, c, i, j} = \sum_{u, v} x_{b,\, c,\, i+u,\, j+v} \cdot w^{\text{dw}}_{c, u, v}
$$

**逐点卷积**（对每个位置做一次通道间的线性变换）：

$$
y_{b, o, i, j} = \sum_{c} W^{\text{pw}}_{o, c} \cdot z_{b, c, i, j}
$$

**参数量对比**（不计偏置）：

| | 参数量 |
| --- | --- |
| 标准卷积 | $C_{\text{out}} \cdot C_{\text{in}} \cdot k^2$ |
| 深度可分离卷积 | $C_{\text{in}} \cdot k^2 + C_{\text{out}} \cdot C_{\text{in}}$ |

两者之比：

$$
\frac{C_{\text{in}} k^2 + C_{\text{out}} C_{\text{in}}}{C_{\text{out}} C_{\text{in}} k^2} = \frac{1}{C_{\text{out}}} + \frac{1}{k^2}
$$

$k = 3$、$C_{\text{out}}$ 较大时约为 $1/9$，参数和计算量都降到原来的九分之一左右。

## 形状

| 量 | 形状 |
| --- | --- |
| `x` | `(B, C_in, H, W)` |
| 窗口 | `(B, C_in, H_out, W_out, kH, kW)` |
| `dw_weight` | `(C_in, 1, kH, kW)` |
| 深度卷积输出 $z$ | `(B, C_in, H_out, W_out)` |
| `pw_weight` | `(C_out, C_in, 1, 1)`，取 `[:, :, 0, 0]` 得 `(C_out, C_in)` |
| 输出 | `(B, C_out, H_out, W_out)` |

$H_{\text{out}} = H - k_H + 1$（无填充、步长 1）。

## 用到的 PyTorch 操作

- `x.unfold(2, kH, 1).unfold(3, kW, 1)`：取出所有窗口。
- `dw_weight[:, 0].view(1, C_in, 1, 1, kH, kW)`：调整形状，让每个通道的核与对应通道的窗口广播相乘。
- `.sum(dim=(-2, -1))`：对窗口内求和。
- `torch.einsum('bchw,oc->bohw', z, W_pw)`：$1 \times 1$ 卷积就是对通道维做矩阵乘。

## 常见坑

- **深度卷积时对通道求和了**：深度卷积的每个通道独立，只对窗口内的空间位置求和。
- **卷积核广播错位**：核的通道维要对齐窗口的通道维（第 1 维）。
- **逐点权重的多余维度**：`(C_out, C_in, 1, 1)` 要去掉最后两个长度为 $1$ 的维度。
- 不能使用 `F.conv2d`。
