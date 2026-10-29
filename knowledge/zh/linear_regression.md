## 背景

线性回归是最简单的机器学习模型：用特征的线性组合预测一个实数。这道题用三种方法求解同一个问题，正好串起机器学习的三种基本思路：解析求解、手推梯度、自动求导。

## 核心公式

**模型与损失**（均方误差）：

$$
\hat{y} = X w + b,
\qquad
L(w, b) = \frac{1}{N} \sum_{i=1}^{N} (x_i^{\top} w + b - y_i)^2 = \frac{1}{N} \lVert X w + b - y \rVert^2
$$

**① 闭式解（正规方程）**：给 $X$ 加一列全 $1$，把偏置并进参数：$\tilde{X} = [X,\ \mathbf{1}]$，$\theta = [w;\ b]$。令梯度为零：

$$
\nabla_\theta L = \frac{2}{N} \tilde{X}^{\top} (\tilde{X}\theta - y) = 0
\quad\Longrightarrow\quad
\tilde{X}^{\top} \tilde{X}\, \theta = \tilde{X}^{\top} y
$$

$$
\theta = (\tilde{X}^{\top} \tilde{X})^{-1} \tilde{X}^{\top} y
$$

实践中不直接求逆（矩阵接近奇异时数值不稳定），而是用最小二乘求解器。

**② 梯度下降**：对 $w$、$b$ 分别求导：

$$
\frac{\partial L}{\partial w} = \frac{2}{N} X^{\top} (X w + b - y),
\qquad
\frac{\partial L}{\partial b} = \frac{2}{N} \sum_{i} (x_i^{\top} w + b - y_i)
$$

每步 $w \leftarrow w - \eta\, \partial L / \partial w$，$b$ 同理。学习率 $\eta$ 太大会发散，太小则收敛很慢。

**③ 自动求导**：只写前向和损失，让 `loss.backward()` 算梯度，`optimizer.step()` 更新参数。

三种方法应得到几乎相同的 $w$、$b$。

## 形状

| 量 | 形状 |
| --- | --- |
| `X` | `(N, D)` |
| `y`、预测、误差 | `(N,)` |
| $\tilde{X}$ | `(N, D + 1)` |
| `w` | `(D,)` |
| `b` | 标量 |
| `nn.Linear(D, 1)` 的权重、输出 | `(1, D)`、`(N, 1)` |

## 用到的 PyTorch 操作

- `torch.cat([X, torch.ones(N, 1)], dim=1)`。
- `torch.linalg.lstsq(A, y).solution`：最小二乘解。
- `nn.Linear`、`nn.MSELoss`、`torch.optim.SGD`、`zero_grad` / `backward` / `step`。

## 常见坑

- **闭式解忘了偏置列**：解出的直线被迫过原点。
- **梯度少了因子 $2/N$**：方向没错，但步长被放大了 $N/2$ 倍，很容易发散。
- **`nn.Linear` 的输出没有 `squeeze`**：`(N, 1)` 和 `(N,)` 传给 `MSELoss` 会被广播成 `(N, N)`，只给一个警告，结果却完全错误。
- **返回形状**：`w` 是 `(D,)`、`b` 是标量；`nn.Linear` 的权重是 `(1, D)`，要去掉第一维。
- 闭式解不能用 autograd。
