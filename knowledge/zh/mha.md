## 背景

单个注意力只能用一种方式衡量「相关」。**多头注意力**（Multi-Head Attention）把特征维切成 $h$ 份，每份独立做一次注意力：有的头可能关注语法关系，有的关注指代，有的关注相邻词。最后把各头的结果拼起来，再用一个线性层融合。

## 核心公式

先用线性层投影，再拆成 $h$ 个头，每个头的维度 $d_k = d_{\text{model}} / h$：

$$
Q' = Q W_Q,\quad K' = K W_K,\quad V' = V W_V
$$

第 $i$ 个头取 $Q', K', V'$ 的第 $i$ 段（每段 $d_k$ 维）做缩放点积注意力：

$$
\text{head}_i = \operatorname{softmax}\!\left(\frac{Q'_i K'^{\top}_i}{\sqrt{d_k}}\right) V'_i
$$

拼接后输出投影：

$$
\operatorname{MHA}(Q, K, V) = \operatorname{Concat}(\text{head}_1, \dots, \text{head}_h)\, W_O
$$

总计算量和单个 $d_{\text{model}}$ 维的注意力差不多，因为每个头只有 $d_{\text{model}}/h$ 维。

## 形状

关键在于把「头」变成一个独立的批维度，这样一次批量矩阵乘就能同时算完所有头：

| 步骤 | 形状 |
| --- | --- |
| 输入 $Q$ | `(B, S_q, d_model)` |
| 投影后 | `(B, S_q, d_model)` |
| `view` 拆头 | `(B, S_q, h, d_k)` |
| `transpose(1, 2)` | `(B, h, S_q, d_k)` |
| 注意力分数 | `(B, h, S_q, S_k)` |
| 各头输出 | `(B, h, S_q, d_k)` |
| `transpose(1, 2)` 再合并 | `(B, S_q, h·d_k) = (B, S_q, d_model)` |

## 用到的 PyTorch 操作

- `nn.Module`：在 `__init__` 里先调用 `super().__init__()`，再把 `nn.Linear` 赋值为属性，参数才会被注册。
- `nn.Linear(d_model, d_model)`。
- `t.view(...)` / `t.reshape(...)`：改变形状，不改变数据顺序。
- `t.transpose(1, 2)`：交换两个维度。
- `t.contiguous()`：`transpose` 后内存不再连续，`view` 之前需要先调用它（或者直接用 `reshape`）。
- `torch.matmul`：对四维张量也按最后两维做矩阵乘。

## 常见坑

- **直接 `view(B, h, S, d_k)`**：形状数字对得上，但数据被错误地切分。必须先 `view(B, S, h, d_k)`，再交换维度。
- **合并时忘了先转置回来**：`(B, h, S, d_k)` 直接 `view(B, S, d_model)` 会把不同头、不同位置的数据混在一起。
- **`transpose` 后 `view` 报错**：先 `.contiguous()`。
- **交叉注意力时 $S_q \ne S_k$**：K、V 的拆头要用它们自己的序列长度。
- **缩放用了 $\sqrt{d_{\text{model}}}$**：应是每个头的维度 $\sqrt{d_k}$。
