## 背景

语言模型的最后一步是把隐状态乘以 LM head 得到整个词表上的 logits，再算交叉熵。这个 logits 张量大得惊人：

$$
N \times V \times 4 \text{ 字节} = (8 \times 4096) \times 128000 \times 4 \approx 16.8 \text{ GB}
$$

（batch 8、序列 4096、LLaMA-3 词表 128K、fp32）。它的梯度还要再占一份。在大词表模型里，这一个张量常常就是激活显存的峰值。

**分块交叉熵**按行切块：每次只算 $c$ 行的 logits，立刻求出这几行的损失并累加，然后丢掉这块 logits。任意时刻只存在一个 $c \times V$ 的块。Liger-Kernel、Apple 的 Cut Cross-Entropy 都是这个思路（它们还会在前向时顺便算出梯度，连反向用的 logits 都不保存）。

## 核心公式

第 $i$ 行的 logits 为 $z_i = h_i W^{\top}$，损失为（log-sum-exp 形式的交叉熵）：

$$
\ell_i = \operatorname{LSE}(z_i) - z_{i, t_i},
\qquad
\operatorname{LSE}(z) = \log \sum_{k} e^{z_k}
$$

**忽略标签**：目标等于 `ignore_index`（通常是 $-100$）的位置不参与损失，比如 padding、或者只训练回答时的提示词部分。设有效行的集合为 $\mathcal{V}$：

$$
L = \frac{1}{|\mathcal{V}|} \sum_{i \in \mathcal{V}} \ell_i
$$

**分块时如何求均值**：必须分别累加「总损失」和「有效行数」，最后再相除：

$$
L = \frac{\sum_{\text{块 } c} \sum_{i \in c \cap \mathcal{V}} \ell_i}{\sum_{\text{块 } c} |c \cap \mathcal{V}|}
$$

不能先算每块的均值再平均：最后一块可能更短，各块的有效行数也不同，「均值的均值」不等于总体均值。

## 形状

| 量 | 形状 |
| --- | --- |
| `hidden` | `(N, D)` |
| `weight` | `(V, D)` |
| 一块 hidden | `(c, D)` |
| 一块 logits | `(c, V)` |
| 一块的 LSE、目标 logit | `(c,)` |

## 用到的 PyTorch 操作

- 切片 `hidden[start:start + c]`，最后一块会自动变短。
- `torch.logsumexp(logits, dim=-1)`：稳定的 LSE。
- `logits.gather(1, idx[:, None])[:, 0]`：每行取出目标列。
- `t.masked_fill(~valid, 0)`：把被忽略的目标换成合法下标，避免 `gather` 越界。
- `hidden.new_zeros(())`：与输入同 dtype、同 device 的标量 $0$。

## 常见坑

- **均值的均值**：按块求平均再平均，在块大小或有效行数不同时结果错误。
- **用 $-100$ 去 `gather`**：越界报错。先换成任意合法下标，再用掩码把这些行的损失乘成 $0$。
- **全部被忽略**：有效行数为 $0$，要返回 $0$ 而不是 $0/0$。
- **一次性算出全部 logits** 再分块：显存峰值没有降下来，测试会检查。
- 梯度要能回传到 `hidden` 和 `weight`，不要用 `.item()` 累加损失。
