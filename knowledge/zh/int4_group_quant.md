## 背景

INT4 量化把每个权重压到 4 位，只有 $16$ 个取值，比 INT8 更激进，所以要用两个技巧把误差压下来：

- **分组**（group-wise）：每一行再切成若干组（常见 $128$ 列一组），每组有自己的量化参数。组越小，组内数值范围越窄，量化越精细。
- **非对称**：用 scale 和 zero point 把组内的实际范围 $[\text{mn}, \text{mx}]$ 映射到 $[0, 15]$，而不是假定范围关于 $0$ 对称。

GPTQ、AWQ 等 4-bit 量化方法都使用这种存储格式，两个 4 位数打包进一个字节。

## 核心公式

**非对称量化**：把区间 $[\text{mn}, \text{mx}]$ 均匀分成 $15$ 格：

$$
s = \frac{\text{mx} - \text{mn}}{15},
\qquad
z = \operatorname{round}\!\left(-\frac{\text{mn}}{s}\right)
$$

$$
q = \operatorname{clamp}\!\left(\operatorname{round}\!\left(\frac{w}{s}\right) + z,\ 0,\ 15\right),
\qquad
\hat{w} = (q - z) \cdot s
$$

zero point $z$ 是实数 $0$ 对应的整数编码。误差不超过半格：$|w - \hat{w}| \le s/2$。

**为什么范围要包含 $0$**：令 $\text{mn} = \min(\min w, 0)$、$\text{mx} = \max(\max w, 0)$。这样 $-\text{mn}/s \in [0, 15]$，$z$ 一定落在合法范围内；而且 $w = 0$ 时 $q = z$，反量化恰好得到 $0$。

**存储开销**：每个权重 4 位，每组额外存一个 scale 和一个 zero。组大小为 $g$、scale 用 32 位、zero 用 8 位时，平均每个权重多出

$$
\frac{32 + 8}{g} \text{ 位},\qquad g = 128 \text{ 时约 } 0.31 \text{ 位}
$$

**打包**：一个字节 8 位，低 4 位放偶数位置的元素，高 4 位放奇数位置的元素：

$$
\text{byte}_i = q_{2i} \;|\; (q_{2i+1} \ll 4)
$$

解包：$q_{2i} = \text{byte}_i \,\&\, \texttt{0xF}$，$q_{2i+1} = \text{byte}_i \gg 4$。

## 形状

| 量 | 形状 |
| --- | --- |
| `w` | `(out, in)` |
| 分组视图 | `(out, in / g, g)` |
| 每组的 mn、mx、scale、zero | `(out, in / g)` |
| 编码 $q$ | `(out, in)`，值在 $[0, 15]$ |
| `packed` | `(out, in / 2)`，`uint8` |

## 用到的 PyTorch 操作

- `w.reshape(out, in // g, g)`：每行连续的 $g$ 列成为一组。
- `t.amin(dim=-1)`、`t.amax(dim=-1)`、`t.clamp(max=0)` / `t.clamp(min=0)`。
- `torch.round`：注意它是**向偶数舍入**（$2.5 \to 2$，$1.5 \to 2$）。
- `t.to(torch.uint8)`，位运算 `|`、`<<` 只能用于整数张量。
- `q[:, 0::2]`、`q[:, 1::2]`：偶数列、奇数列。

## 常见坑

- **在浮点张量上做位运算**：先转成 `uint8`。
- **高低位放反**：偶数位置在低 4 位。
- **范围不包含 0**：全正的组 zero point 会变成负数，被 clamp 后产生大误差。
- **全零组**：$\text{mx} = \text{mn} = 0$，$s = 0$，除以零。此时令 $s = 1$。
- **分组方向错**：组沿每行的列方向（输入维）划分，各行独立。
- 不要修改输入 `w`。
