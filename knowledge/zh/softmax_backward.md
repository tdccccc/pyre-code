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

## 延伸

- **只需要保存输出**：反向公式完全由 $y$ 决定，前向不必保存 $x$。被 mask 掉的位置（如因果注意力的上三角）$y = 0$，对应的 $\mathrm{d}x$ 自动为 $0$，反向里不需要再处理 mask。
- **与交叉熵融合**：softmax 后面接交叉熵时，两者的反向合并成 $y - e_t$（$e_t$ 为目标类别的独热向量），不再需要单独的 softmax 反向，也避开 $1/y_t$ 这一项。见「交叉熵」一题和「手写反向传播」专题。
- **注意力里的写法**：设 $P$ 是注意力权重，$O = PV$，则 $\sum_j \mathrm{d}P_{ij} P_{ij} = \mathrm{d}O_i \cdot O_i$。FlashAttention 的反向借此直接用 $\mathrm{d}O$ 和 $O$ 求出那个点积，不用先物化整行 $P$（见「IO 感知 kernel」专题）。
- **代价**：计算量是 $O(V)$，但每个元素只做一两次乘加，是典型的访存受限算子。一行放得进片上存储时，融合 kernel 读入 $y$ 和 $\mathrm{d}y$ 一次就能算完点积和输出。
- **精度**：点积是对 $V$ 个数求和，输入是 fp16 / bf16 时应在 fp32 下累加。

## 面试追问

1. **softmax 的雅可比矩阵满秩吗？**
   不满秩。$J \mathbf{1} = y - y\,(y^{\top}\mathbf{1}) = 0$，所以 $J$ 奇异，秩至多 $V - 1$。这对应平移不变性：所有 $x_j$ 同加一个常数，输出不变。结果是 $\mathrm{d}x$ 每行之和恒为 $0$，也无法从 $\mathrm{d}x$ 反推出 $\mathrm{d}y$。
2. **softmax 加交叉熵的梯度为什么是 $y - e_t$？**
   此时 $\mathrm{d}y = -e_t / y_t$，代入公式得 $\langle \mathrm{d}y, y\rangle = -1$，于是 $\mathrm{d}x = y \odot (\mathrm{d}y + 1) = y - e_t$。交叉熵的 $1/y_t$ 抵消了雅可比里的 $y_t$。
3. **softmax 饱和时梯度会怎样？**
   输出接近独热时，$y_i (1 - y_i)$ 趋于 $0$，梯度几乎消失。这是注意力里要除以 $\sqrt{d_k}$ 的原因之一：点积太大会让 softmax 饱和，原始 Transformer 论文指出这样会使梯度极小。
4. **如果前向是 $\operatorname{softmax}(x / \tau)$，反向公式怎么变？**
   多乘一个 $1/\tau$：$\mathrm{d}x = \frac{1}{\tau}\, y \odot (\mathrm{d}y - \langle \mathrm{d}y, y\rangle)$，因为链式法则再经过一步缩放。
5. **为什么反向不构造雅可比矩阵？**
   $J$ 每行是 $V \times V$ 的，词表或序列很长时存不下；而 $J^{\top}\mathrm{d}y$ 化简后只要一次点积和一次逐元素运算，是 $O(V)$。VJP 的思路是「只算乘积、不物化矩阵」，所有手写反向都是这样。
