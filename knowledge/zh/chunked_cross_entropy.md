## 背景

语言模型的最后一步是把隐状态乘以 LM head 得到整个词表上的 logits，再算交叉熵。这个 logits 张量大得惊人：

$$
N \times V \times 4 \text{ 字节} = (8 \times 4096) \times 128000 \times 4 \approx 16.8 \text{ GB}
$$

（batch 8、序列 4096、LLaMA-3 词表 128K、fp32）。它的梯度还要再占一份。在大词表模型里，这一个张量常常就是激活显存的峰值。

**分块交叉熵**按行切块：每次只算 $c$ 行的 logits，立刻求出这几行的损失并累加，然后丢掉这块 logits。任意时刻只存在一个 $c \times V$ 的块。Liger-Kernel、Apple 的 Cut Cross-Entropy 都是这个思路。Liger-Kernel 在前向里就按块算出梯度，不保存反向要用的 logits；Cut Cross-Entropy 用自定义 kernel 不把 logits 写进显存，反向时再按块重算。

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

## 延伸

- **梯度也可以分块**：设一块的 logits 为 $z = h W^{\top}$，$z$ 的梯度是 $(p - y)/|\mathcal{V}|$（被忽略的行为 $0$）。于是这一块对两个输入的梯度分别是：

$$
\frac{\partial L}{\partial h} = \frac{\partial L}{\partial z}\, W,
\qquad
\frac{\partial L}{\partial W} \mathrel{+}= \Bigl(\frac{\partial L}{\partial z}\Bigr)^{\top} h
$$

  算完就可以丢掉这块 $z$，$W$ 的梯度在各块之间累加。这就是「前向里顺便算梯度」的做法：不依赖 autograd，也不保存 logits。
- **朴素分块在训练时不一定省显存**：用 Python 循环加 autograd，每块的 logits 都会被保存下来供 `logsumexp` 的反向使用，所有块加起来还是 $N \times V$。本题的显存检查是在 `no_grad` 下进行的，只验证了前向。训练里真正省显存的办法有三种：一是自定义 `autograd.Function`，前向里直接算好梯度（Liger-Kernel）；二是对每块用激活检查点，反向时重算这一块的 logits（见「激活检查点」一题）；三是用融合 kernel，反向时按块重算（Cut Cross-Entropy）。
- **代价**：重算意味着多一次 `hidden @ weight.T`，换取显存。块太小会让矩阵乘的效率变差，块大小是显存和速度的折中。
- **也可以沿词表维分块**：对每个词表块维护一个运行中的最大值和指数和，遇到更大的最大值时把已有的和乘上修正因子，与 Flash Attention 里的在线 softmax 是同一个技巧（见「Flash Attention（分块）」一题）。张量并行时把 LM head 按词表切到多张卡上，各卡先算本地的最大值和指数和，再用通信合并出全局的 LSE（见「分布式训练」专题）。
- **梯度累积下的均值**：每个 micro-batch 的有效 token 数不同，对每个 micro-batch 取均值再平均，同样会出现「均值的均值」问题。需要用整个有效 batch 的 token 总数做分母（见「梯度累积」一题）。
- **`ignore_index`**：PyTorch 的默认值是 $-100$。常见用途有 padding，以及指令微调里只训练回答、不训练提示词的位置。

## 面试追问

1. **为什么大词表模型里 LM head 的 logits 是显存瓶颈？**
   logits 的大小是 $N \times V$，$N$ 是 batch 与序列长度的乘积，$V$ 是词表大小（十万量级）。它比隐状态大 $V / D$ 倍，梯度还要再占一份。
2. **分块时为什么要分别累加总损失和有效行数，最后再相除？**
   各块的有效行数不一样（最后一块更短，有的块里有被忽略的行）。每块先取均值再平均，等于给小块的每一行更大的权重，与全体行直接取均值不同。
3. **用 Python 循环分块，训练时显存真的会降下来吗？**
   不一定。autograd 要保存每块的 logits 给反向使用，累加起来仍是完整的 $N \times V$。要么在前向里就把梯度算出来，要么对每块用检查点，反向时重算。
4. **分块会改变计算结果吗？**
   不会，沿行分块在数学上完全等价，数值上只有浮点求和顺序的微小差别。沿词表维分块需要在线 LSE（运行最大值加修正因子），同样是精确的，不是近似。
5. **所有行都被忽略时怎么处理？**
   有效行数为 $0$，直接除会得到 $0/0$。约定返回 $0$，同时要让它仍然是接在计算图上的标量，这样反向时不会出错。
