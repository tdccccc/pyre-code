## 背景

线性层（全连接层）是神经网络最基本的积木：每个输出都是所有输入的加权和再加一个偏置。注意力里的 Q/K/V 投影、MLP 里的升维降维，全都是线性层。

## 核心公式

对单个输入向量 $x \in \mathbb{R}^{d_{\text{in}}}$：

$$
y = W x + b, \qquad W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}},\ b \in \mathbb{R}^{d_{\text{out}}}
$$

即 $y_j = \sum_{i} W_{j,i}\, x_i + b_j$。实际中输入按行堆叠成矩阵 $X$（每行一个样本），于是写成：

$$
Y = X W^{\top} + b
$$

PyTorch 把权重存成 `(out_features, in_features)`，所以前向要转置。

**初始化**。如果权重方差太大，输出会随层数指数级膨胀；太小则不断衰减。设输入各分量独立、方差为 $1$，则：

$$
\operatorname{Var}(y_j) = d_{\text{in}} \cdot \operatorname{Var}(W_{j,i})
$$

令 $\operatorname{Var}(W) = 1/d_{\text{in}}$，即 $W \sim \mathcal{N}\!\left(0, \tfrac{1}{d_{\text{in}}}\right)$，输出方差就保持在 $1$ 左右。也就是标准正态随机数乘以 $1/\sqrt{d_{\text{in}}}$。偏置初始化为 $0$。

## 形状

| 量 | 形状 |
| --- | --- |
| `x` | `(*, in_features)`，`*` 表示任意多个前导维度 |
| `weight` | `(out_features, in_features)` |
| `bias` | `(out_features,)` |
| 输出 | `(*, out_features)` |

`@` 只作用于最后两维，前导维度原样保留；`bias` 通过广播加到每一行。

## 用到的 PyTorch 操作

- `torch.randn(out, in)`：标准正态随机数。
- `torch.zeros(out)`。
- `t.requires_grad_(True)`，或创建时传 `requires_grad=True`：让参数参与求导。
- `W.T`：二维张量的转置。

## 常见坑

- **权重形状写反**成 `(in, out)`：测试会检查 `weight.shape`。
- **在乘过缩放系数之后的结果上才要求梯度**：先 `randn(...) * scale` 得到新张量，再对它调用 `requires_grad_(True)`。如果先对 `randn` 的结果设 `requires_grad`，再乘系数，存下来的就不是叶子张量了。
- **缩放用了 $1/d_{\text{in}}$**：标准差是 $1/\sqrt{d_{\text{in}}}$，方差才是 $1/d_{\text{in}}$。

## 延伸

- **计算量和参数量**：输入 `(B, S, d_in)`、输出 `(B, S, d_out)` 的线性层，参数量 $d_{\text{in}} d_{\text{out}} + d_{\text{out}}$，前向 FLOPs 约为

$$
2\, B\, S\, d_{\text{in}}\, d_{\text{out}}
$$

  系数 $2$ 是因为一次乘加算两次浮点运算。反向要算 $\partial L/\partial X$ 和 $\partial L/\partial W$ 两个同样大小的矩阵乘，所以反向约为前向的 $2$ 倍，前向加反向合计约 $3$ 倍前向。
- **反向传播公式**：设 $Y = X W^{\top} + b$，损失对 $Y$ 的梯度为 $G = \partial L/\partial Y$，则

$$
\frac{\partial L}{\partial X} = G\, W,
\qquad
\frac{\partial L}{\partial W} = G^{\top} X,
\qquad
\frac{\partial L}{\partial b} = \sum_{\text{行}} G
$$

  其中前导维度（batch、序列）要先展平，$\partial L/\partial W$ 才是对所有 token 求和。
- **为什么权重存成 `(out, in)`**：这是 PyTorch 的约定，前向写作 `x @ W.T`。转置只是改变步长的视图，不会拷贝数据，底层矩阵乘库可以直接按转置方式读取。
- **PyTorch 自带 `nn.Linear` 的默认初始化**：权重和偏置都从 $\mathcal{U}(-1/\sqrt{d_{\text{in}}},\ 1/\sqrt{d_{\text{in}}})$ 均匀采样，方差是 $1/(3 d_{\text{in}})$。题目的写法（标准差 $1/\sqrt{d_{\text{in}}}$ 的高斯、偏置为零）与此不同，更接近经典的 LeCun 初始化；对 ReLU 网络，更准确的缩放是 Kaiming 的 $2/d_{\text{in}}$，见「Kaiming 初始化」一题。
- **大模型里常常没有偏置**：LLaMA 的线性层不带 bias。省下参数和一次加法，对效果几乎没有影响。PaLM 的论文还提到去掉 bias 能让大模型训练更稳定。
- **合并投影**：注意力里的 $W_Q$、$W_K$、$W_V$ 常合并成一个更大的线性层，一次矩阵乘做完再切开，比三次小矩阵乘更能吃满 GPU。
- **算力 vs 访存**：训练和长序列预填充时，矩阵很大，线性层受算力限制；解码阶段每次只有一个 token，相当于矩阵乘向量，每个权重只用一次，受显存带宽限制（见「推理服务」「性能估算」专题）。

## 面试追问

1. **一个 `nn.Linear(d_in, d_out)` 有多少参数、多少 FLOPs？**
   参数 $d_{\text{in}} d_{\text{out}} + d_{\text{out}}$。每个 token 前向约 $2\, d_{\text{in}} d_{\text{out}}$ FLOPs，反向约是前向的两倍。
2. **为什么权重初始化的标准差是 $1/\sqrt{d_{\text{in}}}$？**
   输出是 $d_{\text{in}}$ 项独立随机量之和，方差是 $d_{\text{in}} \operatorname{Var}(W)$。令 $\operatorname{Var}(W) = 1/d_{\text{in}}$，输出方差就与输入相同，信号不会逐层放大或缩小。
3. **两个线性层之间不加激活函数，会怎样？**
   $W_2 (W_1 x) = (W_2 W_1) x$，等价于一个线性层。而且若中间维度 $r$ 比输入输出小，合起来的矩阵秩不超过 $r$，这正是 LoRA 用低秩分解近似权重更新的出发点（见「量化与高效微调」专题）。
4. **写出线性层的反向传播。**
   $\partial L/\partial X = G W$，$\partial L/\partial W = G^{\top} X$，$\partial L/\partial b$ 是 $G$ 按 batch 和序列维度求和。记住 $\partial L/\partial W$ 需要保存前向的输入 $X$。
5. **为什么现在很多大模型的线性层不加偏置？**
   偏置参数很少，对表达能力贡献很小；去掉它能省参数、省一次逐元素加法。LLaMA 等模型的线性层都不带 bias。
