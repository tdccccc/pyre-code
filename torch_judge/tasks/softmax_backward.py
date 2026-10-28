"""Softmax Backward task."""

TASK = {
    "title": "Softmax Backward",
    "title_zh": "Softmax 反向传播",
    "difficulty": "Easy",
    "description_en": "Implement the backward pass of softmax in closed form.\n\nAttention, sampling and classification heads all differentiate through softmax. Backprop needs only the vector-Jacobian product, and the closed form computes it in $O(V)$ per row from the saved output y. The full Jacobian is not needed: for one row, $J = \\operatorname{diag}(y) - y y^\\top$ is a $V \\times V$ matrix, which costs $O(V^2)$ memory and time to build. The product $J^\\top \\texttt{dy}$ never has to be formed, because $(J^\\top \\texttt{dy})_i = y_i \\cdot (\\texttt{dy}_i - \\sum_j \\texttt{dy}_j\\, y_j)$, a single dot product and one elementwise multiply per row.\n\n**Signature:** `softmax_backward(y, dy) -> Tensor`\n\n**Parameters:**\n- `y` — softmax output, shape (*, V), where softmax was taken over the last dim\n- `dy` — upstream gradient $\\mathrm{d}L/\\mathrm{d}y$, same shape as `y`\n\n**Returns:** `dx` = $\\mathrm{d}L/\\mathrm{d}x$, same shape as `y`\n\n**Formula:**\n$$\ndx = y \\odot \\left(dy - \\sum_j dy_j\\, y_j\\right)\n$$\n\n$\\odot$ is elementwise multiplication and the sum runs over the last dimension, kept for broadcasting (in code: `sum(dy * y, dim=-1, keepdim=True)`).\n\n**Constraints:**\n- Closed form only: no autograd (no `torch.autograd`, no `.backward()`), and do not recompute the softmax, since `x` is not given.\n- Do not modify `y` or `dy` in place.\n- Must work for any number of leading dims.",
    "description_zh": "以闭式实现 softmax 的反向传播。\n\n注意力、采样和分类头都需要对 softmax 求导。反向传播只需要向量-雅可比积，而闭式公式利用前向保存的输出 y，每行的计算复杂度为 $O(V)$。不需要完整的雅可比矩阵：对于一行，$J = \\operatorname{diag}(y) - y y^\\top$ 是 $V \\times V$ 的矩阵，构造它需要 $O(V^2)$ 的显存和时间。乘积 $J^\\top \\texttt{dy}$ 无需显式构造，因为 $(J^\\top \\texttt{dy})_i = y_i \\cdot (\\texttt{dy}_i - \\sum_j \\texttt{dy}_j\\, y_j)$，每行只需一次点积和一次逐元素乘法。\n\n**签名:** `softmax_backward(y, dy) -> Tensor`\n\n**参数:**\n- `y` — softmax 输出，形状 (*, V)，softmax 沿最后一维计算\n- `dy` — 上游梯度 $\\mathrm{d}L/\\mathrm{d}y$，与 `y` 形状相同\n\n**返回:** `dx` = $\\mathrm{d}L/\\mathrm{d}x$，与 `y` 形状相同\n\n**公式:**\n$$\ndx = y \\odot \\left(dy - \\sum_j dy_j\\, y_j\\right)\n$$\n\n$\\odot$ 表示逐元素相乘，求和沿最后一维进行，并保留该维以便广播（代码里是 `sum(dy * y, dim=-1, keepdim=True)`）。\n\n**约束:**\n- 只用闭式：不得使用 autograd（不得调用 `torch.autograd` 或 `.backward()`），也不得重新计算 softmax，因为 `x` 不会被传入。\n- 不得原地修改 `y` 或 `dy`。\n- 需支持任意数量的前导维度。",
    "function_name": "softmax_backward",
    "hint": "s = sum(dy * y, dim=-1, keepdim=True)   # one dot product per row\ndx = y * (dy - s)                          # elementwise, O(V) per row",
    "hint_zh": "s = sum(dy * y, dim=-1, keepdim=True)   # 每行一次点积\ndx = y * (dy - s)                          # 逐元素运算，每行 O(V)",
    "tests": [
        {
            "name": "Output shape and dtype",
            "code": """
import torch
torch.manual_seed(0)
y = torch.softmax(torch.randn(4, 5), dim=-1)
dy = torch.randn(4, 5)
dx = {fn}(y, dy)
assert dx.shape == y.shape, f'Expected {tuple(y.shape)}, got {tuple(dx.shape)}'
assert dx.dtype == y.dtype, f'Expected dtype {y.dtype}, got {dx.dtype}'
""",
        },
        {
            "name": "Hand-computed 3-element example",
            "code": """
import torch
y = torch.tensor([0.2, 0.3, 0.5])
dy = torch.tensor([1.0, 0.0, 0.0])
# s = sum(dy * y) = 0.2
# dx = y * (dy - s) = [0.2*0.8, 0.3*(-0.2), 0.5*(-0.2)]
expected = torch.tensor([0.16, -0.06, -0.10])
dx = {fn}(y, dy)
assert torch.allclose(dx, expected, atol=1e-6), f'Expected {expected.tolist()}, got {dx.tolist()}'
""",
        },
        {
            "name": "Matches autograd (2-D)",
            "code": """
import torch
torch.manual_seed(1)
x = torch.randn(6, 7, requires_grad=True)
y = torch.softmax(x, dim=-1)
dy = torch.randn(6, 7)
(ref,) = torch.autograd.grad(y, x, grad_outputs=dy)
dx = {fn}(y.detach(), dy)
assert torch.allclose(dx, ref, atol=1e-5), f'Max diff vs autograd: {(dx - ref).abs().max().item():.2e}'
""",
        },
        {
            "name": "Matches autograd (3-D)",
            "code": """
import torch
torch.manual_seed(2)
x = torch.randn(2, 3, 9, requires_grad=True)
y = torch.softmax(x, dim=-1)
dy = torch.randn(2, 3, 9)
(ref,) = torch.autograd.grad(y, x, grad_outputs=dy)
dx = {fn}(y.detach(), dy)
assert dx.shape == (2, 3, 9), f'Expected (2, 3, 9), got {tuple(dx.shape)}'
assert torch.allclose(dx, ref, atol=1e-5), f'Max diff vs autograd: {(dx - ref).abs().max().item():.2e}'
""",
        },
        {
            "name": "Rows of dx sum to zero",
            "code": """
import torch
torch.manual_seed(3)
for shape in [(5, 11), (2, 4, 6)]:
    y = torch.softmax(torch.randn(*shape), dim=-1)
    dy = torch.randn(*shape)
    dx = {fn}(y, dy)
    row_sums = dx.sum(dim=-1)
    assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-6), f'Rows of dx should sum to 0 for shape {shape}, max |sum| = {row_sums.abs().max().item():.2e}'
""",
        },
        {
            "name": "Row-constant dy gives zero gradient",
            "code": """
import torch
torch.manual_seed(4)
y = torch.softmax(torch.randn(3, 2, 8), dim=-1)
dy = torch.full_like(y, 3.7)  # same upstream value for every element of a row
dx = {fn}(y, dy)
assert torch.allclose(dx, torch.zeros_like(dx), atol=1e-6), f'Expected ~0, max |dx| = {dx.abs().max().item():.2e}'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(5)
y = torch.softmax(torch.randn(4, 6), dim=-1)
dy = torch.randn(4, 6)
y0, dy0 = y.clone(), dy.clone()
{fn}(y, dy)
assert torch.equal(y, y0), 'y was modified in place'
assert torch.equal(dy, dy0), 'dy was modified in place'
""",
        },
        {
            "name": "Works when y does not require grad",
            "code": """
import torch
torch.manual_seed(6)
y = torch.softmax(torch.randn(3, 5), dim=-1).detach()
assert not y.requires_grad
dy = torch.randn(3, 5, requires_grad=True)
dx = {fn}(y, dy)
expected = y * (dy.detach() - (dy.detach() * y).sum(dim=-1, keepdim=True))
assert torch.allclose(dx, expected, atol=1e-6), 'Wrong result when y has requires_grad=False'
""",
        },
    ],
    "solution": '''def softmax_backward(y, dy):
    # Per-row dot product sum_j dy_j * y_j, kept as a broadcastable (*, 1) tensor
    s = (dy * y).sum(dim=-1, keepdim=True)
    # dx_i = y_i * (dy_i - s): the full V x V Jacobian is never formed
    return y * (dy - s)''',
    "demo": """torch.manual_seed(0)
x = torch.randn(2, 5, requires_grad=True)
y = torch.softmax(x, dim=-1)
dy = torch.randn(2, 5)

dx = softmax_backward(y.detach(), dy)
(ref,) = torch.autograd.grad(y, x, grad_outputs=dy)
print('Max diff vs autograd:', (dx - ref).abs().max().item())
print('Row sums of dx:', dx.sum(dim=-1).tolist())""",

}
