## 背景

反向传播时，每个算子要做的事是：已知损失对输出的梯度 $\mathrm{d}y = \partial L / \partial y$，求损失对输入的梯度 $\mathrm{d}x = \partial L / \partial x$。由链式法则，这是一个**向量-雅可比积**（VJP）：

$$
\mathrm{d}x_j = \sum_i \mathrm{d}y_i \,\frac{\partial y_i}{\partial x_j}
\qquad\text{即}\qquad
\mathrm{d}x = J^{\top} \mathrm{d}y
$$

对 softmax 来说，雅可比矩阵是 $V \times V$ 的。词表 $V = 10^5$ 时，每行就要一个 $10^{10}$ 元素的矩阵，根本存不下。好在 softmax 的 VJP 可以化简成 $O(V)$ 的闭式。

## 核心公式

$y_i = e^{x_i} / \sum_k e^{x_k}$。分 $i = j$ 和 $i \ne j$ 两种情况求导，可以统一写成：

$$
\frac{\partial y_i}{\partial x_j} = y_i\,(\delta_{ij} - y_j),
\qquad
J = \operatorname{diag}(y) - y\, y^{\top}
$$

（$\delta_{ij}$ 在 $i = j$ 时为 $1$，否则为 $0$）。$J$ 是对称矩阵。

代入 VJP：

$$
\mathrm{d}x_j = \sum_i \mathrm{d}y_i\, y_i\,(\delta_{ij} - y_j)
= y_j\, \mathrm{d}y_j - y_j \sum_i \mathrm{d}y_i\, y_i
$$

即

$$
\mathrm{d}x = y \odot \bigl(\mathrm{d}y - \langle \mathrm{d}y, y \rangle\bigr),
\qquad
\langle \mathrm{d}y, y \rangle = \sum_i \mathrm{d}y_i\, y_i
$$

每行只需一次点积和一次逐元素运算，$O(V)$。

**两个性质**（测试会检查）：

- 每行梯度之和为 $0$：$\sum_j \mathrm{d}x_j = \langle \mathrm{d}y, y \rangle - \bigl(\sum_j y_j\bigr) \langle \mathrm{d}y, y \rangle = 0$。这对应 softmax 的平移不变性：所有 $x_j$ 同时加一个常数，输出不变。
- 若 $\mathrm{d}y$ 每行是常数 $c$，则 $\langle \mathrm{d}y, y \rangle = c$，$\mathrm{d}x = 0$。

## 形状

`y`、`dy`、`dx` 都是 `(*, V)`；点积 $\langle \mathrm{d}y, y \rangle$ 沿最后一维求和，保留为 `(*, 1)` 以便广播。

## 用到的 PyTorch 操作

- `(dy * y).sum(dim=-1, keepdim=True)`。
- 逐元素乘、减。

## 常见坑

- **构造完整的雅可比矩阵**：结果对，但是 $O(V^2)$ 的时间和显存。
- **忘了 `keepdim`**：点积结果的形状变成 `(*,)`，和 `(*, V)` 相减时广播错位。
- **求和维度错了**：沿最后一维（softmax 的维度）求和。
- **原地修改输入**：如 `dy -= s` 会改掉调用方的张量。
- 不能用 autograd，也拿不到 $x$，只能用 $y$。
