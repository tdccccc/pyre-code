## 背景

模型大到单张 GPU 放不下一层的权重时，就要把**一个矩阵切开**，分到多张卡上各算一部分，这叫**张量并行**。Megatron-LM 给出了 Transformer MLP 的经典切法：第一层按**列**切，第二层按**行**切，整个 MLP 只需要在最后做一次通信。

## 核心公式

MLP：$Y = \operatorname{GELU}(X W_1)\, W_2$，$W_1 \in \mathbb{R}^{d \times d_{ff}}$，$W_2 \in \mathbb{R}^{d_{ff} \times d}$，切成 $n$ 份。

**第一层按列切**：

$$
W_1 = \bigl[\, W_1^{(1)} \ \ W_1^{(2)} \ \cdots \ W_1^{(n)} \,\bigr]
\quad\Rightarrow\quad
X W_1 = \bigl[\, X W_1^{(1)} \ \ \cdots \ \ X W_1^{(n)} \,\bigr]
$$

每张卡拿到完整的 $X$，算出隐藏层的一段。GELU 是**逐元素**的，所以可以各自在本地算：

$$
H^{(i)} = \operatorname{GELU}\bigl(X W_1^{(i)}\bigr)
$$

**第二层按行切**：

$$
W_2 = \begin{pmatrix} W_2^{(1)} \\ \vdots \\ W_2^{(n)} \end{pmatrix}
\quad\Rightarrow\quad
H W_2 = \bigl[\, H^{(1)} \ \cdots \ H^{(n)} \,\bigr] \begin{pmatrix} W_2^{(1)} \\ \vdots \\ W_2^{(n)} \end{pmatrix} = \sum_{i=1}^{n} H^{(i)} W_2^{(i)}
$$

这是分块矩阵乘法。每张卡算出一个完整形状的部分和，最后做一次 **all-reduce（求和）**：

$$
Y = \sum_{i=1}^{n} \underbrace{H^{(i)} W_2^{(i)}}_{\text{第 } i \text{ 张卡的部分结果}}
$$

**为什么不能反过来切**：如果第一层按行切，每张卡只得到 $X W_1$ 的一个部分和，必须先求和才能过 GELU，因为

$$
\operatorname{GELU}(a + b) \ne \operatorname{GELU}(a) + \operatorname{GELU}(b)
$$

这样就要多一次通信。

## 形状

| 量 | 形状 |
| --- | --- |
| `x` | `(B, d_model)` |
| 每个 $W_1^{(i)}$ | `(d_model, d_ff / n)` |
| 每个 $H^{(i)}$ | `(B, d_ff / n)` |
| 每个 $W_2^{(i)}$ | `(d_ff / n, d_model)` |
| 每个部分和、最终输出 | `(B, d_model)` |

## 用到的 PyTorch 操作

- `nn.ParameterList([nn.Parameter(...), ...])`：存放一组参数，会被正确注册。普通 Python 列表里的参数不会被注册。
- `@` 矩阵乘、`torch.erf`（精确 GELU）。
- 用一个循环模拟 $n$ 张卡，把部分结果相加模拟 all-reduce。

## 常见坑

- **先求和再 GELU**，或者对每个分片 GELU 之后再整体 GELU：GELU 只能在第一层之后、各分片本地做一次。
- **把部分结果拼接**（concat）：第二层的部分结果形状都是完整的 `(B, d_model)`，要**相加**。
- **切错方向**：$W_1$ 切的是输出维 $d_{ff}$（列），$W_2$ 切的是输入维 $d_{ff}$（行），两者的切法必须对应。
- **用 Python 列表存分片参数**：参数不会被注册，优化器和测试都找不到。
