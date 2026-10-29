## 背景

分类模型对每个类别输出一个分数（logit），softmax 把它们变成概率。训练目标是让**正确类别的概率尽量大**。交叉熵损失就是正确类别概率的负对数，语言模型的预训练损失（预测下一个 token）也正是它。

## 核心公式

logits $z \in \mathbb{R}^{C}$，正确类别 $y$：

$$
p_c = \frac{e^{z_c}}{\sum_{k} e^{z_k}},
\qquad
\ell = -\log p_y
$$

$p_y \to 1$ 时损失 $\to 0$；$p_y \to 0$ 时损失 $\to \infty$。

展开对数，得到直接从 logits 计算的形式：

$$
\ell = -z_y + \log \sum_{k} e^{z_k} = \operatorname{LSE}(z) - z_y
$$

**log-sum-exp 技巧**：$\operatorname{LSE}$ 直接算会因 $e^{z_k}$ 溢出得到 $\infty$。提出最大值 $m = \max_k z_k$：

$$
\operatorname{LSE}(z) = m + \log \sum_{k} e^{z_k - m}
$$

此时所有指数都 $\le 0$，求和至少为 $1$，既不会上溢，也不会出现 $\log 0$。

**梯度**非常简洁：

$$
\frac{\partial \ell}{\partial z_c} = p_c - \mathbb{1}[c = y]
$$

即「预测概率 − one-hot 标签」。

**batch 平均**：$B$ 个样本的损失取均值 $L = \frac{1}{B} \sum_{n=1}^{B} \ell_n$。

## 形状

| 量 | 形状 |
| --- | --- |
| `logits` | `(B, C)` |
| `targets` | `(B,)`，整数类别 |
| 每个样本的 $\log p$ | `(B, C)` |
| 取出 $\log p_{n, y_n}$ | `(B,)` |
| 输出 | 标量 |

## 用到的 PyTorch 操作

- `torch.logsumexp(z, dim=-1, keepdim=True)`：内置的稳定 LSE。
- **成对索引**：`t[torch.arange(B), targets]` 取出每一行 $n$ 的第 `targets[n]` 列，得到 `(B,)`。
- `.mean()`。

## 常见坑

- **先 softmax 再 `log`**：当 $p_y$ 下溢为 $0$ 时得到 $-\infty$，损失变成 $\infty$ 或 NaN。测试会用很大的 logits 检查。
- **`t[:, targets]`**：这会取出所有行的所有目标列，得到 `(B, B)`，不是每行一个值。
- **用 `sum` 代替 `mean`**：要和 `F.cross_entropy` 的默认行为（取均值）一致。
- **符号**：损失是 $-\log p_y$，为正数。

## 延伸

- **三种等价的说法**：交叉熵、负对数似然（NLL）、one-hot 目标与预测之间的 KL 散度，三者在 one-hot 标签下数值相同。语言模型里常报告的困惑度（perplexity）就是平均交叉熵的指数 $e^{L}$。更完整的关系见「损失函数」专题。
- **初始损失自检**：模型刚初始化、输出接近均匀分布时，损失应当约为 $\log C$。类别数 $C = 1000$ 时约 $6.9$，词表 $32000$ 的语言模型约 $10.4$。训练一开始看到的损失和它差很多，通常说明初始化或标签有问题。
- **为什么直接吃 logits**：PyTorch 的 `F.cross_entropy` 内部是 `log_softmax` 加 `nll_loss`，输入是未归一化的 logits 而不是概率，不要在它前面再接一层 softmax。把两步融合，既稳定（用 LSE），又能直接得到简洁的梯度 $p - y$。
- **精度**：logits 在 bf16 / fp16 下只有几位有效数字，而 LSE 要对很多项求和，所以实际中常把 logits 转成 fp32 再算损失。
- **归约方式**：本题取 batch 均值。语言模型里是对所有有效 token 求均值；有 padding 或被忽略的位置时，分母要用有效 token 数，见「分块 LM Head 交叉熵」一题。
- **反向传播**：不依赖 autograd 手推 softmax 的梯度，见「Softmax 反向传播」一题，属于「手写反向传播」专题。

## 面试追问

1. **为什么 softmax 加交叉熵的梯度是 $p - y$？**
   $\ell = -z_y + \operatorname{LSE}(z)$，第一项对 $z_c$ 的导数是 $-\mathbb{1}[c = y]$，第二项的导数正好是 softmax 概率 $p_c$，相加得 $p_c - \mathbb{1}[c = y]$。
2. **为什么分类用交叉熵而不是 MSE？**
   交叉熵的梯度随预测错误程度线性增大；MSE 作用在 softmax 输出上时多一个 softmax 的导数，模型很自信地答错时这个导数接近 $0$，梯度几乎消失。另外交叉熵对应最大似然，有明确的概率解释。
3. **为什么要用 log-sum-exp 技巧？结果为什么不变？**
   $\operatorname{LSE}(z)$ 对整体平移是等变的：$\operatorname{LSE}(z - m) = \operatorname{LSE}(z) - m$。减去最大值后所有指数 $\le 1$，不会上溢，求和至少为 $1$，$\log$ 也不会遇到 $0$。
4. **模型输出的 logits 全相等时，损失是多少？**
   $\log C$。这是 $C$ 个类别上的均匀分布，每个类别概率都是 $1/C$。
5. **交叉熵和 KL 散度有什么区别？**
   $H(q, p) = H(q) + \operatorname{KL}(q \| p)$。当 $q$ 是固定的目标分布时，$H(q)$ 是常数，两者的梯度相同，只差一个常数。标签是 one-hot 时 $H(q) = 0$，两者数值也相同；标签平滑、蒸馏这类软目标下，二者数值才有差别。
