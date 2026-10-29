## 背景

**自注意力**里 Q、K、V 都来自同一个序列。**交叉注意力**里，查询来自一个序列，键和值来自另一个序列：

- 机器翻译的解码器：Q 来自已生成的译文，K/V 来自编码器对原文的表示
- 文生图模型：Q 来自图像特征，K/V 来自文本编码
- 多模态大模型：Q 来自文本，K/V 来自图像特征

查询序列的每个位置都从另一个序列中「检索」信息。

## 核心公式

设 $X_q \in \mathbb{R}^{S_q \times d}$、$X_{kv} \in \mathbb{R}^{S_{kv} \times d}$：

$$
Q = X_q W_Q, \qquad K = X_{kv} W_K, \qquad V = X_{kv} W_V
$$

$$
\text{head}_i = \operatorname{softmax}\!\left(\frac{Q_i K_i^{\top}}{\sqrt{d_k}}\right) V_i,
\qquad
\text{out} = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

公式和多头自注意力一样，区别只在输入来源。另一个序列的内容是完整给定的，不存在「未来」，所以**不用因果掩码**。

## 形状

| 量 | 形状 |
| --- | --- |
| `x_q` | `(B, S_q, d_model)` |
| `x_kv` | `(B, S_kv, d_model)` |
| 拆头后的 $Q$ | `(B, h, S_q, d_k)` |
| 拆头后的 $K$、$V$ | `(B, h, S_kv, d_k)` |
| 注意力分数 | `(B, h, S_q, S_kv)` |
| 输出 | `(B, S_q, d_model)` |

输出长度跟随**查询**序列，和 $S_{kv}$ 无关。

## 用到的 PyTorch 操作

- `nn.Module`、`nn.Linear(d_model, d_model)` 四个投影。
- `view` + `transpose(1, 2)` 拆头，`transpose` + `contiguous().view` 合头。
- `torch.matmul`、`torch.softmax(..., dim=-1)`。

## 常见坑

- **K/V 拆头用了 $S_q$**：K、V 的序列长度是 $S_{kv}$，两者不相等时会报错或错乱。
- **K/V 投影了错误的输入**：K、V 来自 `x_kv`，Q 来自 `x_q`。
- **加了因果掩码**：交叉注意力里每个查询都应该能看到全部 KV。
- softmax 沿最后一维（$S_{kv}$ 方向）。
