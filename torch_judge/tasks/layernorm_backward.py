"""LayerNorm Backward task."""

TASK = {
    "title": "LayerNorm Backward",
    "title_zh": "LayerNorm 反向传播",
    "difficulty": "Medium",
    "description_en": "Implement the backward pass of LayerNorm by hand.\n\nLayerNorm normalizes each vector over its last dimension, then applies a learned scale and shift. Its gradient is not a term-by-term chain rule: the mean and variance couple all D elements of a row, so dx must subtract both the mean of dx_hat and the projection onto x_hat. Training frameworks implement exactly this fused formula, which avoids storing the intermediate graph.\n\n**Signature:** `layernorm_backward(dy, x, gamma, eps=1e-5) -> tuple[Tensor, Tensor, Tensor]`\n\n**Parameters:**\n- `dy` — upstream gradient, shape (*, D)\n- `x` — layer input, shape (*, D)\n- `gamma` — scale, shape (D,)\n- `eps` — added to the variance inside the square root (default 1e-5)\n\n**Returns:** `(dx, dgamma, dbeta)`, where dx has shape (*, D), and dgamma and dbeta have shape (D,)\n\n**Forward:** $y = (x - \\text{mean})\\cdot\\text{rstd}\\cdot\\gamma + \\beta$, with mean and variance taken over the last dim. The variance is biased (divided by D), and $\\text{rstd} = 1 / \\sqrt{\\text{var} + \\texttt{eps}}$.\n\n**Formula:**\n$$\n\\begin{aligned}\n\\hat{x} &= (x - \\text{mean})\\cdot\\text{rstd} \\\\\nd\\hat{x} &= \\texttt{dy}\\cdot\\gamma \\\\\n\\texttt{dx} &= \\text{rstd}\\cdot(d\\hat{x} - \\operatorname{mean}_{-1}(d\\hat{x}) - \\hat{x}\\cdot\\operatorname{mean}_{-1}(d\\hat{x}\\cdot\\hat{x})) \\\\\n\\texttt{dgamma} &= \\sum_{\\text{leading dims}} \\texttt{dy}\\cdot\\hat{x} \\\\\n\\texttt{dbeta} &= \\sum_{\\text{leading dims}} \\texttt{dy}\n\\end{aligned}\n$$\n\n**Constraints:**\n- Implement manually: no autograd (no `torch.autograd`, no `.backward()`).\n- Means over the last dim keep their dims for broadcasting. The variance divides by D (biased).\n- dgamma and dbeta sum over every dim except the last.\n- Do not modify `dy`, `x` or `gamma` in place.",
    "description_zh": "手工实现 LayerNorm 的反向传播。\n\nLayerNorm 沿最后一维对每个向量做归一化，再施加可学习的缩放和平移。它的梯度不是逐项的链式法则：均值和方差把一行中的全部 D 个元素耦合在一起，因此 dx 必须同时减去 dx_hat 的均值以及它在 x_hat 方向上的投影。训练框架实现的正是这个融合公式，它避免了保存中间计算图。\n\n**签名:** `layernorm_backward(dy, x, gamma, eps=1e-5) -> tuple[Tensor, Tensor, Tensor]`\n\n**参数:**\n- `dy` — 上游梯度，形状 (*, D)\n- `x` — 该层输入，形状 (*, D)\n- `gamma` — 缩放参数，形状 (D,)\n- `eps` — 加在开方内部的方差偏置（默认 1e-5）\n\n**返回:** `(dx, dgamma, dbeta)`，其中 dx 形状为 (*, D)，dgamma 与 dbeta 形状为 (D,)\n\n**前向:** $y = (x - \\text{mean})\\cdot\\text{rstd}\\cdot\\gamma + \\beta$，均值和方差沿最后一维计算。方差为有偏方差（除以 D），$\\text{rstd} = 1 / \\sqrt{\\text{var} + \\texttt{eps}}$。\n\n**公式:**\n$$\n\\begin{aligned}\n\\hat{x} &= (x - \\text{mean})\\cdot\\text{rstd} \\\\\nd\\hat{x} &= \\texttt{dy}\\cdot\\gamma \\\\\n\\texttt{dx} &= \\text{rstd}\\cdot(d\\hat{x} - \\operatorname{mean}_{-1}(d\\hat{x}) - \\hat{x}\\cdot\\operatorname{mean}_{-1}(d\\hat{x}\\cdot\\hat{x})) \\\\\n\\texttt{dgamma} &= \\sum_{\\text{前导维度}} \\texttt{dy}\\cdot\\hat{x} \\\\\n\\texttt{dbeta} &= \\sum_{\\text{前导维度}} \\texttt{dy}\n\\end{aligned}\n$$\n\n**约束:**\n- 手工实现：不得使用 autograd（不得调用 `torch.autograd` 或 `.backward()`）。\n- 沿最后一维的均值保留维度以便广播。方差除以 D（有偏）。\n- dgamma 与 dbeta 对除最后一维以外的所有维度求和。\n- 不得原地修改 `dy`、`x` 或 `gamma`。",
    "function_name": "layernorm_backward",
    "hint": "mu = x.mean(-1, keepdim); var = ((x - mu)**2).mean(-1, keepdim)\nrstd = 1 / sqrt(var + eps); x_hat = (x - mu) * rstd\ndx_hat = dy * gamma\ndx = rstd * (dx_hat - dx_hat.mean(-1, keepdim) - x_hat * (dx_hat * x_hat).mean(-1, keepdim))\ndgamma = (dy * x_hat).sum(leading dims); dbeta = dy.sum(leading dims)",
    "hint_zh": "mu = x.mean(-1, keepdim); var = ((x - mu)**2).mean(-1, keepdim)\nrstd = 1 / sqrt(var + eps); x_hat = (x - mu) * rstd\ndx_hat = dy * gamma\ndx = rstd * (dx_hat - dx_hat.mean(-1, keepdim) - x_hat * (dx_hat * x_hat).mean(-1, keepdim))\ndgamma = (dy * x_hat).sum(除最后一维外的维度); dbeta = dy.sum(除最后一维外的维度)",
    "tests": [
        {
            "name": "Output shapes",
            "code": """
import torch
torch.manual_seed(0)
for shape in [(4, 6), (2, 3, 5)]:
    D = shape[-1]
    x = torch.randn(*shape)
    dy = torch.randn(*shape)
    gamma = torch.randn(D)
    dx, dgamma, dbeta = {fn}(dy, x, gamma)
    assert dx.shape == x.shape, f'dx: expected {tuple(x.shape)}, got {tuple(dx.shape)}'
    assert dgamma.shape == (D,), f'dgamma: expected ({D},), got {tuple(dgamma.shape)}'
    assert dbeta.shape == (D,), f'dbeta: expected ({D},), got {tuple(dbeta.shape)}'
""",
        },
        {
            "name": "Matches autograd through layer_norm (2-D)",
            "code": """
import torch
torch.manual_seed(1)
N, D, eps = 5, 8, 1e-5
x = torch.randn(N, D, requires_grad=True)
gamma = torch.randn(D, requires_grad=True)
beta = torch.randn(D, requires_grad=True)
dy = torch.randn(N, D)
y = torch.nn.functional.layer_norm(x, (D,), gamma, beta, eps)
ref_dx, ref_dg, ref_db = torch.autograd.grad(y, (x, gamma, beta), grad_outputs=dy)
dx, dgamma, dbeta = {fn}(dy, x.detach(), gamma.detach(), eps)
assert torch.allclose(dx, ref_dx, atol=1e-4), f'dx max diff {(dx - ref_dx).abs().max().item():.2e}'
assert torch.allclose(dgamma, ref_dg, atol=1e-4), f'dgamma max diff {(dgamma - ref_dg).abs().max().item():.2e}'
assert torch.allclose(dbeta, ref_db, atol=1e-4), f'dbeta max diff {(dbeta - ref_db).abs().max().item():.2e}'
""",
        },
        {
            "name": "Matches autograd through layer_norm (3-D)",
            "code": """
import torch
torch.manual_seed(2)
B, S, D, eps = 2, 4, 6, 1e-5
x = torch.randn(B, S, D, requires_grad=True)
gamma = torch.randn(D, requires_grad=True)
beta = torch.zeros(D, requires_grad=True)
dy = torch.randn(B, S, D)
y = torch.nn.functional.layer_norm(x, (D,), gamma, beta, eps)
ref_dx, ref_dg, ref_db = torch.autograd.grad(y, (x, gamma, beta), grad_outputs=dy)
dx, dgamma, dbeta = {fn}(dy, x.detach(), gamma.detach(), eps)
assert dx.shape == (B, S, D), f'dx shape {tuple(dx.shape)}'
assert torch.allclose(dx, ref_dx, atol=1e-4), f'dx max diff {(dx - ref_dx).abs().max().item():.2e}'
assert torch.allclose(dgamma, ref_dg, atol=1e-4), f'dgamma must sum over batch and seq: max diff {(dgamma - ref_dg).abs().max().item():.2e}'
assert torch.allclose(dbeta, ref_db, atol=1e-4), f'dbeta must sum over batch and seq: max diff {(dbeta - ref_db).abs().max().item():.2e}'
""",
        },
        {
            "name": "Custom eps is used",
            "code": """
import torch
torch.manual_seed(3)
N, D, eps = 4, 6, 1e-3
# Tiny variance, so eps dominates the denominator
x = (torch.randn(N, D) * 0.02).requires_grad_(True)
gamma = torch.randn(D, requires_grad=True)
beta = torch.zeros(D, requires_grad=True)
dy = torch.randn(N, D)
y = torch.nn.functional.layer_norm(x, (D,), gamma, beta, eps)
(ref_dx,) = torch.autograd.grad(y, x, grad_outputs=dy)
dx, _, _ = {fn}(dy, x.detach(), gamma.detach(), eps)
assert torch.allclose(dx, ref_dx, atol=1e-5), f'eps=1e-3 not honored: max diff {(dx - ref_dx).abs().max().item():.2e}'
""",
        },
        {
            "name": "Hand-computed 3-element example",
            "code": """
import math, torch
x = torch.tensor([[1.0, 2.0, 3.0]])
dy = torch.tensor([[1.0, 0.0, 0.0]])
gamma = torch.ones(3)
# mean = 2, biased var = 2/3, eps = 0 so rstd = sqrt(3/2)
# x_hat = [-1, 0, 1] * rstd, dx_hat = dy
# dx = rstd * ([1, 0, 0] - 1/3 - x_hat * (-rstd/3)) = rstd * [1/6, -1/3, 1/6]
r = math.sqrt(1.5)
expected = torch.tensor([[r / 6, -r / 3, r / 6]])
dx, dgamma, dbeta = {fn}(dy, x, gamma, 0.0)
assert torch.allclose(dx, expected, atol=1e-5), f'Expected {expected.tolist()}, got {dx.tolist()}'
# dgamma = dy * x_hat = [-rstd, 0, 0]; dbeta = dy = [1, 0, 0]
assert torch.allclose(dgamma, torch.tensor([-r, 0.0, 0.0]), atol=1e-5), f'dgamma wrong: {dgamma.tolist()}'
assert torch.allclose(dbeta, torch.tensor([1.0, 0.0, 0.0]), atol=1e-6), f'dbeta wrong: {dbeta.tolist()}'
""",
        },
        {
            "name": "Rows of dx sum to zero when gamma is all ones",
            "code": """
import torch
torch.manual_seed(4)
x = torch.randn(3, 4, 10)
dy = torch.randn(3, 4, 10)
gamma = torch.ones(10)
dx, _, _ = {fn}(dy, x, gamma)
row_sums = dx.sum(dim=-1)
assert torch.allclose(row_sums, torch.zeros_like(row_sums), atol=1e-5), f'max |row sum| = {row_sums.abs().max().item():.2e}'
""",
        },
        {
            "name": "Float64 matches autograd tightly",
            "code": """
import torch
torch.manual_seed(5)
N, D, eps = 6, 9, 1e-5
x = torch.randn(N, D, dtype=torch.float64, requires_grad=True)
gamma = torch.randn(D, dtype=torch.float64, requires_grad=True)
beta = torch.zeros(D, dtype=torch.float64, requires_grad=True)
dy = torch.randn(N, D, dtype=torch.float64)
y = torch.nn.functional.layer_norm(x, (D,), gamma, beta, eps)
ref_dx, ref_dg, ref_db = torch.autograd.grad(y, (x, gamma, beta), grad_outputs=dy)
dx, dgamma, dbeta = {fn}(dy, x.detach(), gamma.detach(), eps)
assert dx.dtype == torch.float64, f'dx dtype should stay float64, got {dx.dtype}'
assert torch.allclose(dx, ref_dx, atol=1e-10, rtol=0), f'dx max diff {(dx - ref_dx).abs().max().item():.2e}'
assert torch.allclose(dgamma, ref_dg, atol=1e-10, rtol=0), f'dgamma max diff {(dgamma - ref_dg).abs().max().item():.2e}'
assert torch.allclose(dbeta, ref_db, atol=1e-10, rtol=0), f'dbeta max diff {(dbeta - ref_db).abs().max().item():.2e}'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(6)
x = torch.randn(4, 7)
dy = torch.randn(4, 7)
gamma = torch.randn(7)
x0, dy0, g0 = x.clone(), dy.clone(), gamma.clone()
{fn}(dy, x, gamma)
assert torch.equal(x, x0), 'x was modified in place'
assert torch.equal(dy, dy0), 'dy was modified in place'
assert torch.equal(gamma, g0), 'gamma was modified in place'
""",
        },
    ],
    "solution": '''def layernorm_backward(dy, x, gamma, eps=1e-5):
    D = x.shape[-1]
    mu = x.mean(dim=-1, keepdim=True)
    var = ((x - mu) ** 2).mean(dim=-1, keepdim=True)       # biased variance
    rstd = 1.0 / torch.sqrt(var + eps)
    x_hat = (x - mu) * rstd

    dx_hat = dy * gamma
    dx = rstd * (dx_hat
                 - dx_hat.mean(dim=-1, keepdim=True)
                 - x_hat * (dx_hat * x_hat).mean(dim=-1, keepdim=True))

    # Parameter grads sum over every dim except the last
    lead = tuple(range(dy.dim() - 1))
    dgamma = (dy * x_hat).sum(dim=lead)
    dbeta = dy.sum(dim=lead)
    return dx, dgamma, dbeta''',
    "demo": """torch.manual_seed(0)
x = torch.randn(2, 3, 6, requires_grad=True)
gamma = torch.randn(6, requires_grad=True)
beta = torch.zeros(6, requires_grad=True)
dy = torch.randn(2, 3, 6)

y = torch.nn.functional.layer_norm(x, (6,), gamma, beta, 1e-5)
ref_dx, ref_dg, ref_db = torch.autograd.grad(y, (x, gamma, beta), grad_outputs=dy)
dx, dgamma, dbeta = layernorm_backward(dy, x.detach(), gamma.detach())

print('dx max diff:', (dx - ref_dx).abs().max().item())
print('dgamma max diff:', (dgamma - ref_dg).abs().max().item())
print('dbeta max diff:', (dbeta - ref_db).abs().max().item())""",

}
