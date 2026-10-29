## 背景

用 one-hot 标签训练时，交叉熵要求 $p_y \to 1$，这只有在正确类别的 logit 比其他类别高出无穷多时才能达到。模型于是会不断放大 logits，变得过度自信，泛化变差。

**标签平滑**把一小部分概率 $\varepsilon$ 从正确类别「匀」给其他类别，目标分布不再是 one-hot，最优的 logits 就是有限的。原始 Transformer 用 $\varepsilon = 0.1$。

## 核心公式

$C$ 个类别，正确类别 $y$，平滑后的目标分布：

$$
q_c =
\begin{cases}
1 - \varepsilon, & c = y \\[1mm]
\dfrac{\varepsilon}{C - 1}, & c \ne y
\end{cases}
\qquad \sum_c q_c = 1
$$

损失是目标分布与预测分布的交叉熵：

$$
\ell = -\sum_{c=1}^{C} q_c \log p_c
= (1 - \varepsilon)\bigl(-\log p_y\bigr) + \frac{\varepsilon}{C - 1} \sum_{c \ne y} \bigl(-\log p_c\bigr)
$$

$\varepsilon = 0$ 时 $q$ 就是 one-hot，退化为普通交叉熵。

**最优解**：交叉熵在 $p = q$ 时最小，所以模型被鼓励输出 $p_y = 1 - \varepsilon$，而不是 $1$，对应的 logit 差是有限值：

$$
z_y - z_c = \log \frac{1 - \varepsilon}{\varepsilon / (C - 1)}
$$

$\log p$ 用稳定的 log-softmax 计算：$\log p_c = z_c - \operatorname{LSE}(z)$，$\operatorname{LSE}(z) = m + \log \sum_k e^{z_k - m}$。

## 形状

| 量 | 形状 |
| --- | --- |
| `logits`、$\log p$、$q$ | `(N, C)` |
| `targets` | `(N,)` |
| 每个样本的损失 | `(N,)` |
| 输出 | 标量（均值） |

## 用到的 PyTorch 操作

- `torch.full_like(t, value)`：同形状、填满某个值。
- `t.scatter_(dim, index, value)`：按索引原地写值。`soft.scatter_(1, targets.unsqueeze(1), 1 - eps)` 把每行第 `targets[n]` 列设为 $1 - \varepsilon$；`index` 必须和 `t` 维数相同，所以要 `unsqueeze(1)` 成 `(N, 1)`。
- `torch.log_softmax(logits, dim=-1)` 或手写稳定版本。
- `.sum(dim=-1).mean()`。

## 常见坑

- **分母用 $C$**：本题非正确类别分到 $\varepsilon / (C - 1)$。注意 PyTorch 内置的 `F.cross_entropy(label_smoothing=ε)` 用的是另一种约定：所有类别（包括正确类别）都加 $\varepsilon / C$，正确类别为 $1 - \varepsilon + \varepsilon / C$，两者数值不同。
- **`scatter_` 的 index 维数不对**：`targets` 是一维的，要先 `unsqueeze(1)`。
- **不稳定的 $\log(\operatorname{softmax})$**：大 logits 时得到 $-\infty$，乘以 $q_c$ 后变成 NaN。
- 先对类别求和、再对样本求均值。

## 延伸

- **梯度**：目标是 $q$ 时，对 logits 的梯度是 $p - q$（见「损失函数」专题）。普通交叉熵里，$p_y < 1$ 总有梯度，推着 logits 一直增大；标签平滑下 $p = q$ 时梯度为 $0$，logits 的差停在有限值。
- **损失的下界不再是 $0$**：最小值是目标分布的熵 $H(q)$，即使模型完美拟合了 $q$，损失也不为零：

$$
H(q) = -(1 - \varepsilon)\log(1 - \varepsilon) - \varepsilon \log\frac{\varepsilon}{C - 1}
$$

  所以平滑后的损失数值不能和普通交叉熵直接比较，也不能用「损失接近 $0$」判断训练好了。
- **等价的写法**：把 one-hot 与均匀分布做混合，$q = (1 - \varepsilon')\,\text{onehot} + \varepsilon'\, u$，$u$ 是均匀分布。展开后损失是 $(1-\varepsilon')$ 倍的普通交叉熵，加上 $\varepsilon'$ 倍的「预测分布与均匀分布的交叉熵」，后一项把预测拉向均匀。PyTorch 内置的就是这种混合写法，和本题的参数换算是 $\varepsilon' = \varepsilon \cdot \frac{C}{C - 1}$。
- **对校准的影响**：标签平滑通常能让模型的置信度更接近真实正确率，减轻过度自信。但 Müller 等人（2019）也指出，它会让同一类样本的倒数第二层表示挤得更紧，丢掉类间相似度信息，使这样训练出的模型当教师做知识蒸馏时效果变差。
- **对困惑度的影响**：原始 Transformer 论文提到，标签平滑会使困惑度变差（模型学得更「不确定」），但翻译的准确率和 BLEU 有提升。所以它的收益要用任务指标衡量，不能看损失。
- **类别数很大时**：非目标类每个只分到 $\varepsilon / (C - 1)$，数值很小，但总量仍是 $\varepsilon$，整体的「拉向均匀」作用依然在。

## 面试追问

1. **标签平滑为什么能缓解过度自信？**
   one-hot 目标要求 $p_y \to 1$，只有 logit 差趋于无穷才能达到，模型就会不停放大 logits。平滑后目标里 $p_y = 1 - \varepsilon$，最优的 logit 差是有限值，$z_y - z_c = \log\frac{1 - \varepsilon}{\varepsilon / (C - 1)}$。
2. **标签平滑后，训练损失的最小值是多少？**
   是目标分布的熵 $H(q)$，不是 $0$。模型输出恰好等于 $q$ 时达到。
3. **标签平滑对知识蒸馏有什么影响？**
   教师用标签平滑训练，输出的类间相似度信息会变少，同类样本的表示被压得更紧，学生从它的软输出里学到的东西比普通教师少，蒸馏效果往往变差。
4. **标签平滑和温度缩放（temperature scaling）都能改善校准，区别是什么？**
   标签平滑发生在训练时，改变了模型学到的 logits；温度缩放是训练后的事后处理，在验证集上拟合一个标量温度，不改变预测的类别，只缩放置信度。
