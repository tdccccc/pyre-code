## 解决什么问题

训练就是对损失 $L$ 求每个参数的梯度。网络是一串算子的复合，**链式法则**把总梯度拆成每个算子各自的局部梯度。反向传播（reverse-mode 自动微分）从 $L$ 出发，沿计算图倒着走：每个算子收到输出的梯度 $\mathrm{d}y = \partial L / \partial y$，交出输入的梯度 $\mathrm{d}x = \partial L / \partial x$。

这一步是一个**向量-雅可比积**（VJP, vector-Jacobian product）：

$$
\mathrm{d}x = J^{\top}\,\mathrm{d}y,
\qquad
J_{ij} = \frac{\partial y_i}{\partial x_j}
$$

关键在于：**只算乘积 $J^{\top}\mathrm{d}y$，不构造 $J$**。输出和输入各有 $n$ 个元素时，$J$ 有 $n^2$ 个，根本存不下；而几乎所有深度学习算子的 $J$ 都有结构（对角、低秩加对角、分块），乘积可以化简成 $O(n)$ 的闭式。手写反向传播练的就是这种化简。

为什么用反向而不是正向：损失是标量，参数有几亿个。反向模式一遍就能得到所有参数的梯度；正向模式（JVP）每遍只能求一个方向上的导数，要求全部梯度得跑和参数个数同量级的遍数。

通用步骤：

1. 把前向写成一串基本步骤，每步的输入输出明确。
2. 倒序走，每一步用该步的 VJP，把 $\mathrm{d}y$ 变成 $\mathrm{d}x$。
3. 一个量被多处使用（分支、残差、共享参数）时，各处传回来的梯度**相加**。
4. 想清楚反向要用前向的哪些量，前向时保存它们，或者准备好重算（见下文）。

## 对比

按算子的类型，梯度有几种典型形式：

| 类型 | 例子 | 梯度的形式 | 前向要保存 |
| --- | --- | --- | --- |
| 逐元素 | ReLU、GELU、exp | $\mathrm{d}x = \mathrm{d}y \odot f'(\cdot)$ | 输入，或能算出 $f'$ 的输出 |
| 矩阵乘 | Linear、注意力里的 $QK^{\top}$ | $\mathrm{d}X = \mathrm{d}Y\,W^{\top}$，$\mathrm{d}W = X^{\top}\mathrm{d}Y$ | 两个操作数 |
| 归约 | sum、mean、max | 把 $\mathrm{d}y$ 铺回原形状；mean 再除以 $n$；max 只给取到最大值的位置 | 形状；max 的位置 |
| 广播 | $\gamma$ 乘到每一行 | 对被广播的维度**求和** | 形状 |
| 归一化 | softmax、LayerNorm、BatchNorm | 减均值项、减投影项，再缩放 | 输出，或 $x$ 加统计量 |

几点说明：

- **逐元素**最简单，雅可比是对角阵，没有跨元素的耦合。sigmoid 和 tanh 的导数能直接用输出表示（$y(1-y)$ 和 $1-y^2$），所以保存输出就够了；GELU 的导数要用到输入。
- **广播和归约互为对偶**：前向把一个量复制到多处，反向就把多处的梯度加起来；前向把多个量加成一个，反向就把梯度复制回去。LayerNorm 里 $\gamma$ 被每一行共享，所以 $\mathrm{d}\gamma$ 要对所有行求和，正是这个对偶。
- **归一化类**的特殊在于：输出的每个元素依赖整个组的所有输入元素，雅可比是稠密的，但稠密部分是低秩的，所以仍然可以化简。

## 归一化类的共同形式

三个例子，$\mathrm{d}x$ 都是「从上游梯度里减掉几个方向上的分量，再缩放」。

softmax（$y$ 是概率分布，$\langle \mathrm{d}y, y\rangle$ 是 $\mathrm{d}y$ 按 $y$ 加权的均值）：

$$
\mathrm{d}x = y \odot \bigl(\mathrm{d}y - \langle \mathrm{d}y, y\rangle\bigr)
$$

LayerNorm（$\hat{x}$ 是归一化后的值，$r = 1/\sqrt{\sigma^2+\epsilon}$，$\mathrm{d}\hat{x} = \mathrm{d}y \odot \gamma$）：

$$
\mathrm{d}x = r\Bigl(\mathrm{d}\hat{x} - \operatorname{mean}(\mathrm{d}\hat{x}) - \hat{x}\odot\operatorname{mean}(\mathrm{d}\hat{x}\odot\hat{x})\Bigr)
$$

RMSNorm 没有减均值，所以少一项：

$$
\mathrm{d}x = r\Bigl(\mathrm{d}\hat{x} - \hat{x}\odot\operatorname{mean}(\mathrm{d}\hat{x}\odot\hat{x})\Bigr)
$$

BatchNorm 训练模式的形式和 LayerNorm 完全一样，只是两个 $\operatorname{mean}$ 沿 batch（和空间）维度而不是特征维度。

这个形式有清楚的几何含义：前向「抹掉」了哪些自由度，反向就必须让梯度在这些方向上为 $0$。

- softmax 对平移不变，所以 $\mathrm{d}x$ 与全 $1$ 向量正交，每行之和为 $0$。
- LayerNorm 对平移和缩放不变（$\epsilon$ 可忽略时），所以 $\mathrm{d}x$ 同时与全 $1$ 向量和 $\hat{x}$ 正交。
- RMSNorm 只对缩放不变，所以只有 $\hat{x}$ 这一项。

这也给了一个很好的自检：写完公式，先看它的输出是否满足这些正交性质。

## softmax 与交叉熵的梯度为什么这么简洁

分类损失是 softmax 接交叉熵。设 $y = \operatorname{softmax}(x)$，目标类别为 $t$：

$$
L = -\log y_t,
\qquad
\mathrm{d}y = -\frac{1}{y_t}\,e_t
$$

（$e_t$ 是第 $t$ 位为 $1$ 的独热向量）。代入 softmax 的 VJP，$\langle \mathrm{d}y, y\rangle = -1$：

$$
\mathrm{d}x = y \odot \bigl(\mathrm{d}y + 1\bigr) = y - e_t
$$

简洁的原因：交叉熵里的 $1/y_t$ 恰好抵消 softmax 雅可比里的 $y_t$。结果是「预测概率减目标分布」，每个分量都落在 $[-1, 1]$，没有除法，不会因为 $y_t$ 很小而爆炸。

- 目标是软标签 $q$（各分量非负、和为 $1$，如标签平滑）时，结果是 $y - q$。
- batch 取平均时，再除以样本数。
- 工程上要从 logits 直接算（见「交叉熵」一题和「损失函数」专题）：先 softmax 得到 $y$、再单独算 $1/y_t$ 的做法，在 $y_t$ 下溢时会得到无穷大。

## 显存与重计算

反向要用前向的中间量，保存它们的显存占用常常比参数还大。每个算子都有一个选择：保存，还是反向时重算。

| 算子 | 保存 | 为什么够用 |
| --- | --- | --- |
| softmax | 输出 $y$ | $J$ 完全由 $y$ 决定，不需要 $x$ |
| 融合的 LayerNorm | 输入 $x$，每行的 $\mu$ 和 $r$ | $\hat{x}$ 可以用它们廉价地重算 |
| FlashAttention | 输出 $O$，每行的 $\log\sum\exp$ | 注意力矩阵 $P$ 在反向里分块重算 |

经验法则：**逐元素和归一化类的中间量重算很便宜**（本来就是访存受限，多算几个乘法几乎不花时间），可以放心丢；**矩阵乘的结果重算很贵**，通常保存。

FlashAttention 是这个取舍最极端的例子。朴素实现要保存 $P = \operatorname{softmax}(S)$，形状 `(B, H, S, S)`，显存随序列长度平方增长。FlashAttention 不保存 $P$，反向时从 $Q$、$K$ 和保存的 $\log\sum\exp$ 分块重算它，用多出的计算换掉了 $O(S^2)$ 的显存和相应的显存读写（细节见「IO 感知 kernel」专题）。

softmax 反向里那个点积项在这里也有了更省的写法。设 $S$ 是 softmax 之前的打分，$O = PV$，$\mathrm{d}P = \mathrm{d}O\,V^{\top}$：

$$
\sum_j \mathrm{d}P_{ij}\,P_{ij} = \mathrm{d}O_i \cdot O_i
$$

$$
\mathrm{d}S_{ij} = P_{ij}\,\bigl(\mathrm{d}P_{ij} - \mathrm{d}O_i \cdot O_i\bigr)
$$

也就是说，softmax 反向要的 $\langle \mathrm{d}y, y\rangle$ 可以直接由 $\mathrm{d}O$ 和 $O$ 得到，不必先把整行 $P$ 和 $\mathrm{d}P$ 拿出来做点积。

同一个取舍放大到整层，就是激活检查点（「优化与训练技巧」专题）：只保存少数检查点，其余在反向时重跑前向。

## 怎么自检

手写反向最容易出错，且错了往往不报错，只是训练变差。几种从强到弱的检查：

**1. 对照 autograd。** 同一输入上分别用自己的公式和 `torch.autograd.grad` 求梯度，比较。本站的测试就是这么做的。

**2. 有限差分（gradcheck）。** 用中心差分近似每个偏导：

$$
\frac{\partial f}{\partial x_k} \approx \frac{f(x + h e_k) - f(x - h e_k)}{2h}
$$

误差是 $O(h^2)$。PyTorch 自带：

```python
from torch.autograd import gradcheck
x = torch.randn(4, 8, dtype=torch.float64, requires_grad=True)
gradcheck(fn, (x,), eps=1e-6, atol=1e-5)
```

要点：**必须用 float64**，float32 的舍入误差会淹没差分结果；`fn` 里要走你自己写的反向（比如包成 `torch.autograd.Function`），否则只是在拿 autograd 检查 autograd。

**3. 点积检验。** 不想逐元素比，可以取随机的 $v$ 和 $\mathrm{d}y$，检查 $\langle \mathrm{d}y, J v\rangle = \langle J^{\top}\mathrm{d}y, v\rangle$：左边用差分 $\bigl(f(x+hv)-f(x-hv)\bigr)/2h$ 近似 $Jv$，右边用你的反向。一次前向对两遍就能测整个算子。

**4. 性质检验。** 检查上一节讲的正交性质：softmax 的 $\mathrm{d}x$ 每行和为 $0$；LayerNorm 的 $\mathrm{d}x$ 每行和也为 $0$，对任意 $\gamma$ 都成立，因为减均值项把常数方向整个去掉了。这类检查便宜，能抓出漏项。

## 演进

1. **逐层手推**：早期的深度学习框架里，每一层都要自己实现 `forward` 和 `backward`，梯度公式要手推。
2. **自动微分**：现代框架记录前向的计算图，按图自动生成反向，用户只写前向。代价是图里每个中间量都可能被保存。
3. **融合 kernel**：LayerNorm、softmax、交叉熵这类访存受限的算子，把前向和反向各写成一个 kernel，只保存必要的少量量（如 $\mu$、$r$），避免来回读写显存。
4. **用重算换显存**：FlashAttention 不保存 $P$，激活检查点不保存整段激活，都是主动把一部分中间量改成反向时重算。

## 怎么选

默认交给 autograd。自己写 `backward`（`torch.autograd.Function`）的典型理由：

- **融合 kernel**：要把一串算子合成一个，autograd 看不到内部结构，必须自己给反向。
- **省显存**：自动生成的反向会保存太多中间量，手写版本只保存少量统计量，或者选择重算。
- **数值稳定**：把 softmax 和交叉熵合成一个算子，从 logits 直接求 $y - e_t$，比分开算稳。
- **不可微或需要特殊梯度的算子**：例如量化里的直通估计器（「量化与高效微调」专题），前向取整、反向把梯度直接放过去。

## 题目顺序

1. **Softmax 反向传播**：最小的例子。雅可比是「对角阵减一个外积」，化简成一次点积加一次逐元素运算，练「不构造雅可比」这个核心思路。
2. **LayerNorm 反向传播**：同样的思路用到更复杂的算子上：三条依赖路径（直接、经过均值、经过方差）要合并，还要处理广播对应的求和（$\mathrm{d}\gamma$、$\mathrm{d}\beta$）。

## 面试追问

1. **为什么深度学习用反向模式自动微分，而不是正向模式？**
   损失是标量、参数极多。反向模式一遍就得到全部参数的梯度，代价是前向的常数倍；正向模式每遍只给出一个方向上的导数，要得到全部梯度需要的遍数和参数个数同量级。代价在于反向要保存前向的中间量，所以显存成了瓶颈。
2. **为什么 softmax 加交叉熵的梯度是 $y - e_t$？**
   交叉熵 $-\log y_t$ 的梯度里有 $1/y_t$，softmax 的雅可比里有 $y_t$，两者抵消，只剩「预测概率减目标分布」。它还说明：直接从 logits 融合着算，比先算概率再取对数更简单也更稳。
3. **前向做了广播，反向要做什么？举个例子。**
   对被广播的维度求和。例如 $\gamma$ 形状是 `(D,)`，前向被广播到 `(B, S, D)` 的每一行；反向 $\mathrm{d}\gamma$ 要对 $B$、$S$ 两个维度都求和。归约和广播互为对偶，所有「共享参数」的梯度都是这样。
4. **FlashAttention 的反向为什么重算 $P$，而不是保存？**
   $P$ 的大小是 $O(S^2)$，保存和读回的显存读写成本很高；重算只需要 $Q$、$K$ 和每行的 $\log\sum\exp$，多出的计算量换来显存和读写量的大幅减少。标准注意力的瓶颈主要在访存而不是计算，所以这笔交换划算。
5. **手写了反向传播，怎么验证对不对？**
   先用 float64 做 `gradcheck`（中心差分），再与 autograd 对照，最后检查结构性质（如 softmax 的 $\mathrm{d}x$ 每行和为 $0$）。要特别注意 float32 下差分基本不可用，以及测试里必须真的走到自己写的反向。
