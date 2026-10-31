## 背景

Transformer 每层的前馈网络（FFN）负责对每个 token 单独做非线性变换，参数量约占整个模型的三分之二。LLaMA 系列用 SwiGLU 门控结构代替了「Linear → GELU → Linear」：两路投影，一路过 SiLU 后当作门，逐元素控制另一路。

这道题把它写成一个标准的 `nn.Module`，和实际模型代码的组织方式一样。

## 核心公式

$$
\operatorname{SiLU}(z) = z \cdot \sigma(z), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

$$
\operatorname{FFN}(x) = W_{\text{down}}\Bigl(\operatorname{SiLU}(W_{\text{gate}}\, x) \odot W_{\text{up}}\, x\Bigr)
$$

- $W_{\text{gate}}, W_{\text{up}}$：$d_{\text{model}} \to d_{\text{ff}}$
- $W_{\text{down}}$：$d_{\text{ff}} \to d_{\text{model}}$
- $\odot$：逐元素乘

**参数量**：三个矩阵共 $3\, d_{\text{model}}\, d_{\text{ff}}$，而普通 FFN 只有两个矩阵，为 $2\, d_{\text{model}}\, d_{\text{ff}}$。为了保持总参数量相当，LLaMA 把 $d_{\text{ff}}$ 取为约 $\tfrac{2}{3} \cdot 4 d_{\text{model}} = \tfrac{8}{3} d_{\text{model}}$。

## 形状

| 量 | 形状 |
| --- | --- |
| `x` | `(*, d_model)` |
| `gate_proj(x)`、`up_proj(x)` | `(*, d_ff)` |
| 输出 | `(*, d_model)` |

`nn.Linear` 只作用在最后一维，前导维度任意。

## 用到的 PyTorch 操作

- 继承 `nn.Module`，`__init__` 开头调用 `super().__init__()`。
- `nn.Linear(in, out)`：带偏置的线性层，计算 $x W^{\top} + b$。
- `torch.sigmoid`，或 `F.silu` / `nn.SiLU`。
- `forward(self, x)`：定义前向，调用时写 `module(x)`。

## 常见坑

- **属性名不对**：题目要求 `gate_proj`、`up_proj`、`down_proj`，测试会按名字访问。
- **SiLU 用错一路**：只对 `gate_proj(x)` 激活，`up_proj(x)` 保持线性。
- **漏了 `super().__init__()`**：给 `self` 赋值子模块时会报错。

## 延伸

- **在块里的位置**：一个 Pre-Norm 的 Transformer 块里，FFN 和注意力平级，各自套一层「归一化 → 子层 → 残差相加」：

$$
x \leftarrow x + \operatorname{FFN}\bigl(\operatorname{Norm}(x)\bigr)
$$

  归一化见「归一化」专题，整个块的拼装见「GPT-2 Block」一题。
- **逐 token 独立**：FFN 对序列里每个位置用同一组权重各算各的，不在 token 之间交换信息；跨 token 的信息混合全部由注意力完成。因此 FFN 容易并行，也与 KV Cache 无关（缓存的是注意力的 $K$、$V$）。
- **参数占比**：取 $d_{\text{ff}} = \tfrac{8}{3} d$ 时，FFN 每层有 $8 d^2$ 个参数。多头注意力的四个投影共 $4 d^2$，所以 FFN 约占一层参数的三分之二。使用 GQA 等减少 KV 头数的方案后，注意力的参数更少，FFN 占比更大。
- **一种解读**：有研究（Geva 等，2021）把 FFN 看成键值记忆：升维矩阵的每一行像一个「键」，检测输入里的某种模式；降维矩阵对应的列像「值」，决定检测到之后往残差流里写什么。这是一种帮助理解的视角，不是严格结论。
- **偏置**：本题用带偏置的 `nn.Linear`；LLaMA 一类模型通常设 `bias=False`，没有偏置。测试检查的是所有参数都收到梯度，两种写法都能通过。
- **初始化**：`down_proj` 直接写入残差流，大模型里常把它的初始化标准差额外缩小，让残差流方差不随层数增长，见「基础层与激活」专题。
- **和 MoE 的关系**：混合专家模型把这一个 FFN 换成多个并行的 FFN（专家），每个 token 只经过其中几个，见「模型架构」专题。
- **激活内存**：反向需要用到 `gate_proj(x)` 和 `up_proj(x)` 两个大小为 `(B, S, d_ff)` 的中间结果，是训练时激活内存的主要来源之一。激活重计算可以只存输入、反向时再算（见「优化与训练技巧」专题）。

## 面试追问

1. **FFN 在 Transformer 里起什么作用？**
   对每个 token 独立做非线性变换。注意力负责在 token 之间搬运信息，FFN 负责对搬来的信息做加工，同时承载了模型大部分参数。
2. **LLaMA 的 FFN 是什么结构？参数量是多少？**
   SwiGLU：$\operatorname{down}\bigl(\operatorname{SiLU}(\operatorname{gate}(x)) \odot \operatorname{up}(x)\bigr)$，三个矩阵共 $3\, d\, d_{\text{ff}}$ 个参数，$d_{\text{ff}}$ 约取 $\tfrac{8}{3} d$，所以每层约 $8 d^2$。
3. **FFN 的参数量和注意力相比如何？**
   取 $d_{\text{ff}} = 4d$ 的普通 FFN 或 $\tfrac{8}{3}d$ 的 SwiGLU 都是 $8d^2$，多头注意力的 Q、K、V、O 四个投影共 $4d^2$，所以 FFN 约占三分之二。
4. **FFN 为什么不在 token 之间交互？这对推理有什么影响？**
   同一组权重逐 token 独立计算，只有注意力混合 token。所以解码时新 token 的 FFN 只需要自己的输入，不依赖历史 token，历史相关的缓存只和注意力有关。
5. **`gate_proj` 和 `up_proj` 能合并成一个矩阵吗？**
   可以。沿输出维拼接成 $d \to 2 d_{\text{ff}}$ 的一个线性层，一次矩阵乘后再切成两半，数学上完全等价，但一次大矩阵乘比两次小矩阵乘更能吃满 GPU。
