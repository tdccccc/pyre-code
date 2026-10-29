## 背景

全注意力中每个位置看所有位置，计算和显存都是 $O(S^2)$，序列一长就吃不消。**滑动窗口注意力**让每个位置只看自己附近 $w$ 范围内的位置，把代价降到 $O(S \cdot w)$。Longformer、Mistral 都用到了它。

虽然单层只看得到局部，但层层堆叠后视野会扩大：第 $1$ 层每个位置看到 $\pm w$，第 $2$ 层看到的是「邻居的邻居」，$L$ 层之后的感受野约为 $\pm L \cdot w$，和卷积网络的感受野增长方式一样。

## 核心公式

$$
\text{scores}_{ij} =
\begin{cases}
\dfrac{q_i \cdot k_j}{\sqrt{d_k}}, & |i - j| \le w \\[2mm]
-\infty, & |i - j| > w
\end{cases}
\qquad
\text{out} = \operatorname{softmax}(\text{scores})\, V
$$

$S = 5$、$w = 1$ 时，可见位置（$\cdot$）形成一条对角带：

$$
\begin{pmatrix}
\cdot & \cdot & \times & \times & \times \\
\cdot & \cdot & \cdot & \times & \times \\
\times & \cdot & \cdot & \cdot & \times \\
\times & \times & \cdot & \cdot & \cdot \\
\times & \times & \times & \cdot & \cdot
\end{pmatrix}
$$

两个极端：$w = 0$ 时每个位置只看自己，输出就是 $V$；$w \ge S - 1$ 时等同于全注意力。

## 形状

`Q`、`K`、`V` 为 `(B, S, D)`；距离矩阵和掩码为 `(S, S)`，`unsqueeze(0)` 后广播到分数的 `(B, S, S)`。

## 用到的 PyTorch 操作

- `idx = torch.arange(S)`，`(idx.unsqueeze(0) - idx.unsqueeze(1)).abs()`：$|i - j|$ 矩阵。
- 比较运算 `> w` 得到布尔掩码。
- `scores.masked_fill(mask, float('-inf'))`、`torch.softmax`、`torch.bmm`。

## 常见坑

- **边界差一**：窗口包含 $|i - j| = w$，屏蔽的是 $|i - j| > w$，不是 $\ge w$。
- **用 0 代替 $-\infty$**：被屏蔽的位置仍会分到权重。
- 本题窗口是**对称**的（左右都看）；Mistral 里的是因果滑动窗口，只看左边，不要混淆。
- 注意本题实现仍会构造完整的 $S \times S$ 分数矩阵，只是演示掩码规则；真正省内存的实现要按块只计算带内的部分。
