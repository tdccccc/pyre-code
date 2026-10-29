## 背景

普通数据并行中，每个 rank 都保存完整的参数、梯度和 Adam 状态。Adam 的两个动量 $m$、$v$（fp32）加上 fp32 主权重，占了模型状态的绝大部分（见「训练显存估算」一题）。

**ZeRO-2** 把梯度和优化器状态切成 $W$ 份：rank $r$ 只负责参数向量的第 $r$ 段，只保存这一段的梯度、$m$、$v$，也只更新这一段。最后再把各段更新后的参数收集起来，每个 rank 重新拥有完整参数。

## 核心公式

**为什么可以切**：Adam 的更新是**逐元素**的，每个参数的新值只依赖它自己的梯度和动量：

$$
m_i \leftarrow \beta_1 m_i + (1 - \beta_1) g_i,
\qquad
v_i \leftarrow \beta_2 v_i + (1 - \beta_2) g_i^2,
\qquad
p_i \leftarrow p_i - \eta \frac{m_i / (1 - \beta_1^t)}{\sqrt{v_i / (1 - \beta_2^t)} + \epsilon}
$$

所以把元素任意分组、各组独立更新，结果和整体更新完全一样。

**一步的三个阶段**（$N$ 个参数，$W$ 个 rank，分区大小 $n = N/W$）：

1. **reduce-scatter**：rank $r$ 得到第 $r$ 段的**平均**梯度

$$
g^{(r)} = \frac{1}{W} \sum_{k=1}^{W} \text{grad}_k\bigl[\, rn : (r+1)n \,\bigr]
$$

   取平均是因为全局损失是全局 batch 上的均值，每个 rank 的梯度只是其中一份数据的梯度。

2. **本地 Adam**：rank $r$ 用 $g^{(r)}$ 和自己保存的 $m^{(r)}$、$v^{(r)}$ 更新第 $r$ 段参数。

3. **all-gather**：把 $W$ 段新参数按顺序拼接。

**显存**：每个 rank 的优化器状态和梯度降为原来的 $1/W$。**通信**：reduce-scatter + all-gather 的总量和一次 all-reduce 相同，没有额外开销。

## 形状

| 量 | 形状 |
| --- | --- |
| `flat_param`、每个 `rank_grads[k]` | `(N,)` |
| 每个分区的 $g$、$m$、$v$、新参数段 | `(N / W,)` |
| 返回的完整参数 | `(N,)` |

## 用到的 PyTorch 操作

- 切片 `t[lo:hi]`。
- `torch.stack([...]).mean(dim=0)`：多个 rank 的梯度取平均。
- `torch.cat(parts)`：模拟 all-gather。
- `t.sqrt()`。

## 常见坑

- **梯度求和而不是平均**：等价于学习率放大 $W$ 倍，与在平均梯度上运行的 Adam 对不上。
- **原地修改 $m$、$v$ 或参数**：题目要求返回新张量，输入保持不变。
- **偏差校正的步数**：用传入的 `step`，不要自己计数。
- **状态没有传回**：返回的状态在下一步要原样传入，布局必须和输入一致（每个 rank 一个 `{"m", "v"}` 字典）。
- $\epsilon$ 加在 $\sqrt{\hat v}$ 之外。
