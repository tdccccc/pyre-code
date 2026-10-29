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
