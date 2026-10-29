## 背景

Transformer 处理的是 token 序列。**Vision Transformer**（ViT，论文标题是 *An Image is Worth 16x16 Words*）的做法很直接：把图像切成不重叠的小方块（patch），每块展平成一个向量，再用线性层投影成嵌入，这样一张图像就变成了一个 token 序列。

## 核心公式

图像尺寸 $H \times W$、通道数 $C$，块大小 $P$：

$$
N = \frac{H}{P} \cdot \frac{W}{P}
$$

每块展平后的长度是 $C \cdot P \cdot P$，投影为 $d$ 维嵌入：

$$
z_n = E\, \operatorname{flatten}(\text{patch}_n) + b,
\qquad E \in \mathbb{R}^{d \times C P^2}
$$

**例子**：$224 \times 224$ 的 RGB 图像，$P = 16$：$N = 14 \times 14 = 196$ 个块，每块 $3 \times 16 \times 16 = 768$ 维。

**等价于卷积**：这个操作和「卷积核大小 = 步长 = $P$、输出通道为 $d$」的卷积完全等价。很多实现直接用 `nn.Conv2d(C, d, kernel_size=P, stride=P)`。

## 形状

关键在于把每个块的像素正确地聚到一起：

| 步骤 | 形状 |
| --- | --- |
| 输入 | `(B, C, H, W)` |
| 拆分高和宽 | `(B, C, n_h, P, n_w, P)`，$H = n_h P$，$W = n_w P$ |
| 调整维度顺序 | `(B, n_h, n_w, C, P, P)` |
| 合并 | `(B, n_h · n_w, C · P · P)` |
| 线性投影 | `(B, N, embed_dim)` |

块按行优先排列：先第一行的所有块，从左到右，再第二行。每块内部按通道、行、列的顺序展平。

## 用到的 PyTorch 操作

- `x.reshape(B, C, n_h, P, n_w, P)`：把 $H$ 拆成「第几块 × 块内第几行」，$W$ 同理。
- `x.permute(0, 2, 4, 1, 3, 5)`：把两个块编号维放到前面，块内维度放到后面。
- `.reshape(B, N, C * P * P)`：`permute` 之后内存不连续，用 `reshape` 而不是 `view`。
- `nn.Linear(C * P * P, embed_dim)`。

## 常见坑

- **不做 `permute` 直接 `reshape(B, N, C·P·P)`**：形状对了，但每个「块」里混进了不同位置的像素。
- **展平顺序不对**：要按 $(C, P, P)$ 的顺序展平，测试会用手动投影逐块比对。
- **忘了保存 `self.num_patches`**：测试会访问这个属性。
- **`permute` 后用 `view` 报错**：换成 `reshape`，或先 `.contiguous()`。
