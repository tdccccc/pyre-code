## 解决什么问题

几乎所有神经网络都是由少数几种积木搭起来的：

- **线性层**（Linear）：把向量投影到另一个空间，承担网络里绝大部分参数和计算。
- **激活函数**：插在线性层之间引入非线性。没有它，多层线性层合起来仍然是一层。
- **嵌入层**（Embedding）：把离散的 token 编号变成向量，是语言模型的入口。
- **Dropout**：训练时随机丢弃一部分激活，起正则化作用。
- **初始化**：决定训练开始时信号和梯度的尺度，是上面所有层能否训起来的前提。

前馈网络（FFN / MLP）是这些积木最典型的组合。普通写法和门控写法分别是：

$$
\operatorname{FFN}(x) = W_2\, \phi(W_1 x),
\qquad
\operatorname{GatedFFN}(x) = W_{\text{down}}\bigl(\phi(W_{\text{gate}}\, x) \odot W_{\text{up}}\, x\bigr)
$$

$\phi$ 是激活函数，$\odot$ 是逐元素乘。这个专题里的激活函数（ReLU、GELU、SwiGLU）就是在改 $\phi$ 和这个结构，初始化与 Dropout 管的是同一个网络怎么训得稳、训得不过拟合。

## 对比

三种激活函数的差异：

| 方法 | 公式 | 平滑 | 额外参数 | 典型使用 |
| --- | --- | --- | --- | --- |
| ReLU | $\max(0, x)$ | 否，$0$ 处有拐点 | 无 | 原始 Transformer、CNN |
| GELU | $x\,\Phi(x)$ | 是 | 无 | BERT、GPT-2、ViT |
| SwiGLU | $(xW_{\text{up}}) \odot \operatorname{SiLU}(xW_{\text{gate}})$ | 是 | 多一个投影矩阵 | LLaMA、PaLM、Qwen、Mistral |

几个性质：

- ReLU 在负半轴输出恒为 $0$，梯度也为 $0$；GELU 和 SiLU 在负半轴有一小段负值，所以不是单调函数（GELU 最低约 $-0.17$，SiLU 最低约 $-0.28$）。
- ReLU 和 GELU 是单路的 $\phi(x)$；SwiGLU 是两路相乘，激活只作用在门控那一路。
- 同一结构里 GELU 换成 SiLU 差别很小，真正的结构变化是「门控」。

其余积木的角色：

| 积木 | 参数量 | 一句话 |
| --- | --- | --- |
| Linear | $d_{\text{in}} d_{\text{out}}$（外加偏置） | 稠密投影，计算量的主体 |
| Embedding | $V \cdot d$ | 查表，等价于 one-hot 乘矩阵，计算量几乎为零 |
| 门控 MLP | $3\, d\, d_{\text{ff}}$ | 三个 Linear，每层参数量的大头 |
| Dropout | 无 | 只在训练时起作用 |

## 演进

1. **ReLU 取代 sigmoid / tanh**：sigmoid 的导数最大只有 $1/4$，层数一深梯度就连乘到接近 $0$；ReLU 在正半轴导数恒为 $1$，计算也只是一次比较，成了深层网络的默认选择。
2. **GELU（2016）**：用平滑的「软开关」代替硬拐点，BERT、GPT 系列之后成为 Transformer 的标准激活。
3. **Swish / SiLU**：$x\,\sigma(x)$，形状和 GELU 很接近，实现只需要一个 sigmoid。
4. **门控 FFN（Shazeer，2020）**：比较了 GLU 的多种变体（ReGLU、GEGLU、SwiGLU 等），在参数量相当时，这些门控结构的效果优于普通 FFN。PaLM、LLaMA 采用 SwiGLU 之后，它成为大模型 FFN 的主流写法。
5. **初始化**：Xavier（2010）为对称激活推导，Kaiming（2015）针对 ReLU 修正，大 Transformer 再叠加按层数缩放残差分支的做法（见下文）。
6. **Dropout（2014）**：曾经几乎每个模型都用；大模型预训练时代，数据量大、训练通常不过一轮，Dropout 常被关掉（见下文）。

## 门控 MLP：为什么隐藏维约是 $\tfrac{8}{3} d$

普通 FFN 有两个矩阵，隐藏维通常取 $4d$：

$$
\text{参数量} = 2 \cdot d \cdot 4d = 8 d^2
$$

门控 FFN 有三个矩阵。要让总参数量（和计算量）与普通 FFN 持平，隐藏维 $d_{\text{ff}}$ 要满足：

$$
3\, d\, d_{\text{ff}} = 8 d^2
\quad\Longrightarrow\quad
d_{\text{ff}} = \tfrac{8}{3}\, d
$$

所以门控 FFN 并不是「多花一个矩阵的钱」，而是在预算不变的前提下，把一个宽的隐藏层换成两个窄一点的并行投影。实际模型还会把 $d_{\text{ff}}$ 向上取整到某个数（如 $256$）的倍数，方便 GPU 计算。例如 LLaMA-7B 的 $d = 4096$，$\tfrac{8}{3} d \approx 10923$，实际取的是 $11008$。

## 初始化与方差保持

训练刚开始时，我们希望每一层输出的方差不要逐层漂移。对线性层 $y_j = \sum_{i=1}^{n_{\text{in}}} W_{j,i}\, x_i$，设权重均值为 $0$、与输入独立：

$$
\operatorname{Var}(y_j) = n_{\text{in}} \cdot \operatorname{Var}(W) \cdot \mathbb{E}[x^2]
$$

反向传播同理：梯度传回输入时是对 $n_{\text{out}}$ 个输出求和，需要 $n_{\text{out}} \operatorname{Var}(W) \approx 1$。两条要求不能同时满足，不同方法取了不同的折中：

| 方法 | $\operatorname{Var}(W)$ | 针对的激活 | 要保持的是 |
| --- | --- | --- | --- |
| LeCun / 只看 fan_in | $1/n_{\text{in}}$ | 近似线性 | 前向方差 |
| Xavier（Glorot） | $2/(n_{\text{in}} + n_{\text{out}})$ | tanh 等对称激活 | 前向与反向的折中 |
| Kaiming（He） | $2/n_{\text{in}}$ | ReLU | 前向方差（`mode='fan_in'`） |

Kaiming 的因子 $2$ 来自 ReLU：它把一半的输入清零，$\mathbb{E}[\operatorname{ReLU}(h)^2] = \tfrac12 \operatorname{Var}(h)$，权重方差要翻倍才能补回来。不补的话，每过一层能量减半，$30$ 层之后只剩 $2^{-30}$ 左右，信号已经消失。

深层 Transformer 还有一个额外问题：残差流上每个子层都会加一份输出。若各分支独立，残差流的方差大致随层数线性增长。GPT-2 的做法是把残差分支最后一个投影的初始化缩小，缩放因子大致是层数的平方根的倒数，使整条残差流在初始化时仍保持在同一量级。

## Embedding 与 Linear 的关系

Embedding 就是输入为 one-hot 的 Linear（无偏置）：

$$
\operatorname{Embed}(i) = e_i^{\top} W
$$

直接查表省掉了乘以大量 $0$ 的开销，前向只读一行，反向也只更新被查到的那几行。

反过来，语言模型最后一层把隐状态映射到词表的 logits：

$$
\text{logits} = h\, E^{\top}, \qquad E \in \mathbb{R}^{V \times d}
$$

这是一个 $d \to V$ 的 Linear，它的权重形状和输入 Embedding 的权重一样，都是 `(V, d)`。**权重绑定**（weight tying）就是让两者共用同一个矩阵 $E$：

- **省参数**：$V \cdot d$ 在小模型里占比很大，例如 GPT-2 small 的词表 $50257$、$d = 768$，这一个矩阵就有约 $3.86 \times 10^7$ 个参数。
- **语义一致**：「输入时这个词的表示」和「输出时预测这个词的方向」使用同一组向量。
- **代价**：输入和输出被迫共用一套表示，限制了灵活性。模型越大，$V \cdot d$ 占总参数的比例越小，绑定带来的好处也越小，很多大模型选择不绑定。

## Dropout 为什么在大模型预训练里常被关掉

Dropout 的作用是抑制过拟合。传统训练里数据集小、要反复训练多个 epoch，模型很容易记住训练集，Dropout 很有用。大模型预训练的情形反过来：

- 语料远大于模型能记住的量，训练通常不过一轮，每个样本只见一次，没有「记住训练集」的问题。
- 此时模型的瓶颈是**欠拟合**。Dropout 给每一步的梯度加噪声，会让收敛变慢，等于白白浪费算力。
- 正则化的需求转移到了别处：权重衰减（weight decay）、数据去重与质量过滤。

Dropout 并没有消失，它回到了数据少、容易过拟合的场合：小数据集上的微调（如 LoRA 的 dropout 选项）、视觉模型里的 DropPath、较小的模型或多轮训练。

## 怎么选

- **激活函数**：新做 LLM 的 FFN 用 SwiGLU；复现 BERT、GPT-2、ViT 这类已有结构，就跟它原来的激活函数走（GELU），否则预训练权重对不上；CNN 或对延迟很敏感的小模型用 ReLU 就够了。
- **初始化**：用 ReLU 系激活时用 Kaiming；tanh 或线性用 Xavier；Transformer 按所参考模型的惯例（如标准差 $0.02$，残差分支再缩小）。有归一化层时对初始化没那么敏感，但深层网络仍然值得认真设置。
- **Embedding**：小模型可以考虑权重绑定省参数；大模型一般独立。
- **Dropout**：预训练大模型默认不开；小数据微调、过拟合明显时再开，$p$ 取较小的值（如 $0.1$）。

## 题目顺序

1. **线性回归**：最简单的模型，把解析解、手推梯度、自动求导三种思路放在一起，先建立「线性映射 + 损失 + 梯度」的直觉。
2. **线性层**：从回归里的一个线性映射推广到 `nn.Linear`，引入权重形状与初始化缩放。
3. **ReLU**：最简单的激活函数，认识逐元素运算和梯度的流动。
4. **GELU**：换成平滑的软开关，认识 erf 与精确形式、近似形式的区别。
5. **SwiGLU**：从单路激活进入门控，用函数形式写出两路相乘的 FFN。
6. **SwiGLU MLP**：把 SwiGLU 封装成 `nn.Module`，和实际模型代码的组织方式对应。
7. **Embedding**：模型入口，把 Linear 的 one-hot 视角换成查表。
8. **Dropout**：第一次遇到训练与推理两种模式。
9. **权重初始化**：回头用方差分析说明前面这些层为什么要那样缩放。

## 面试追问

1. **为什么神经网络需要非线性激活？**
   没有激活函数，多层线性层合起来等价于一层：$W_2 (W_1 x) = (W_2 W_1) x$。非线性让网络能拟合复杂函数。
2. **ReLU、GELU、SwiGLU 的演进是怎样的，为什么大模型改用 SwiGLU？**
   ReLU 简单高效但有硬拐点、负半轴梯度为零；GELU 换成平滑门控；SwiGLU 进一步引入门控分支，两路相乘。Shazeer 的实验显示在参数量相当时门控结构的效果更好，PaLM、LLaMA 采用后成为主流。目前对它为什么有效，主要是经验结论。
3. **SwiGLU 的隐藏维为什么取约 $\tfrac{8}{3} d$？**
   三个矩阵、隐藏维 $d_{\text{ff}}$ 时参数量是 $3\, d\, d_{\text{ff}}$，普通 FFN 隐藏维 $4d$ 时是 $8 d^2$，令两者相等得 $d_{\text{ff}} = \tfrac{8}{3} d$。
4. **Xavier 和 Kaiming 初始化有什么区别？**
   两者都是让每层输出的方差保持不变。Xavier 为 tanh 等对称激活设计，取 $2/(n_{\text{in}} + n_{\text{out}})$，兼顾前向和反向；Kaiming 针对 ReLU 会清零一半输入的特点，取 $2/n_{\text{in}}$。
5. **Embedding 和 Linear 是什么关系？什么是权重绑定？**
   Embedding 是输入为 one-hot 的无偏置 Linear，只是用查表实现。语言模型的输出头是 $d \to V$ 的 Linear，形状与输入 Embedding 相同，让两者共享同一个矩阵就是权重绑定，能省下 $V \cdot d$ 个参数。
6. **为什么大模型预训练经常不用 Dropout？**
   预训练数据量大，通常不过一轮，模型处在欠拟合而不是过拟合的区间，Dropout 引入的噪声只会拖慢收敛。过拟合主要靠数据规模和 weight decay 来控制。
