"""GAT Layer (Graph Attention) task."""

TASK = {
    "title": "GAT Layer (Graph Attention)",
    "title_zh": "GAT 层（图注意力）",
    "difficulty": "Medium",
    "description_en": "Implement a single-head Graph Attention Network (GAT) layer.\n\nGAT computes attention coefficients between connected nodes, then aggregates neighbor features weighted by those coefficients.\n\n**Signature:** `gat_layer(A, X, W, a, negative_slope=0.2) -> Tensor`\n\n**Parameters:**\n- `A` — adjacency matrix (N, N), binary, already includes self-loops\n- `X` — node feature matrix (N, F_in)\n- `W` — weight matrix (F_in, F_out)\n- `a` — attention vector (2*F_out,)\n- `negative_slope` — LeakyReLU negative slope (default 0.2)\n\n**Returns:** updated node features (N, F_out)\n\n**Algorithm:**\n1. Project features: $H = XW$ → (N, F_out)\n2. Compute attention scores: $e_{ij} = \\operatorname{LeakyReLU}\\left(a_1^\\top H_i + a_2^\\top H_j\\right)$, where $a_1$ = `a[:F_out]` and $a_2$ = `a[F_out:]`\n3. Mask non-edges: set $e_{ij} = -\\infty$ where $A_{ij} = 0$\n4. Normalize over neighbors: $\\alpha_{ij} = \\operatorname{softmax}_j(e_{ij})$ (`softmax(e, dim=-1)`)\n5. Aggregate: $\\text{output} = \\alpha H$\n\n**Hint:** Compute e efficiently as an (N, N) matrix using broadcasting:\n`e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`",
    "description_zh": "实现单头图注意力网络（GAT）层。\n\nGAT 计算相连节点之间的注意力系数，然后用这些系数对邻居特征进行加权聚合。\n\n**签名:** `gat_layer(A, X, W, a, negative_slope=0.2) -> Tensor`\n\n**参数:**\n- `A` — 邻接矩阵 (N, N)，二值，已包含自环\n- `X` — 节点特征矩阵 (N, F_in)\n- `W` — 权重矩阵 (F_in, F_out)\n- `a` — 注意力向量 (2*F_out,)\n- `negative_slope` — LeakyReLU 负斜率（默认 0.2）\n\n**返回:** 更新后的节点特征 (N, F_out)\n\n**算法:**\n1. 特征投影：$H = XW$ → (N, F_out)\n2. 计算注意力分数：$e_{ij} = \\operatorname{LeakyReLU}\\left(a_1^\\top H_i + a_2^\\top H_j\\right)$，其中 $a_1$ = `a[:F_out]`，$a_2$ = `a[F_out:]`\n3. 掩码非边：在 $A_{ij} = 0$ 处令 $e_{ij} = -\\infty$\n4. 对邻居归一化：$\\alpha_{ij} = \\operatorname{softmax}_j(e_{ij})$（`softmax(e, dim=-1)`）\n5. 聚合：$\\text{output} = \\alpha H$\n\n**提示:** 利用广播高效计算 e 矩阵：\n`e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`",
    "function_name": "gat_layer",
    "hint": "Compute e as (N,N) via broadcasting: `e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`, then mask with A and softmax.",
    "hint_zh": "利用广播计算 e 矩阵：`e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)`，然后用 A 掩码并 softmax。",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(42)
N, F_in, F_out = 5, 8, 4
A = torch.ones(N, N)
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
a = torch.randn(2 * F_out)
out = {fn}(A, X, W, a)
assert out.shape == (N, F_out), f'Shape mismatch: {out.shape} vs {(N, F_out)}'
""",
        },
        {
            "name": "Attention weights sum to 1 (via output verification)",
            "code": """
import torch
torch.manual_seed(0)
N, F_in, F_out = 4, 6, 3
A = torch.tensor([[1,1,0,0],[1,1,1,0],[0,1,1,1],[0,0,1,1]], dtype=torch.float)
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
a = torch.randn(2 * F_out)
out = {fn}(A, X, W, a)
# Verify output matches expected (which implies correct attention normalization)
H = X @ W
F_o = F_out
e = (H @ a[:F_o]).unsqueeze(1) + (H @ a[F_o:]).unsqueeze(0)
e = torch.nn.functional.leaky_relu(e, 0.2)
e = e.masked_fill(A == 0, float('-inf'))
alpha = torch.softmax(e, dim=-1)
expected = alpha @ H
assert torch.allclose(out, expected, atol=1e-5), f'Output mismatch — attention normalization may be wrong'
""",
        },
        {
            "name": "Exact numerical value (small graph)",
            "code": """
import torch
torch.manual_seed(123)
N, F_in, F_out = 3, 2, 2
A = torch.tensor([[1,1,0],[1,1,1],[0,1,1]], dtype=torch.float)
X = torch.tensor([[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]])
W = torch.tensor([[0.5, -0.5],[0.5, 0.5]])
a = torch.tensor([1.0, 0.0, 0.0, 1.0])
out = {fn}(A, X, W, a, negative_slope=0.2)
H = X @ W
F_o = F_out
e = (H @ a[:F_o]).unsqueeze(1) + (H @ a[F_o:]).unsqueeze(0)
e = torch.nn.functional.leaky_relu(e, 0.2)
e = e.masked_fill(A == 0, float('-inf'))
alpha = torch.softmax(e, dim=-1)
ref = alpha @ H
assert torch.allclose(out, ref, atol=1e-5), f'Value mismatch:\\n{out}\\nvs\\n{ref}'
""",
        },
        {
            "name": "Gradient flow",
            "code": """
import torch
torch.manual_seed(42)
N, F_in, F_out = 4, 6, 3
A = torch.ones(N, N)
X = torch.randn(N, F_in, requires_grad=True)
W = torch.randn(F_in, F_out, requires_grad=True)
a = torch.randn(2 * F_out, requires_grad=True)
out = {fn}(A, X, W, a)
out.sum().backward()
assert X.grad is not None, 'X.grad is None'
assert W.grad is not None, 'W.grad is None'
assert a.grad is not None, 'a.grad is None'
""",
        },
        {
            "name": "Different graph sizes",
            "code": """
import torch
torch.manual_seed(7)
for N in [2, 8, 16]:
    F_in, F_out = 4, 3
    A = (torch.rand(N, N) > 0.5).float()
    A = A + torch.eye(N)
    A = (A > 0).float()
    X = torch.randn(N, F_in)
    W = torch.randn(F_in, F_out)
    a = torch.randn(2 * F_out)
    out = {fn}(A, X, W, a)
    assert out.shape == (N, F_out), f'Shape mismatch for N={N}: {out.shape}'
""",
        },
        {
            "name": "Custom negative_slope",
            "code": """
import torch
torch.manual_seed(99)
N, F_in, F_out = 3, 4, 2
A = torch.ones(N, N)
X = torch.randn(N, F_in)
W = torch.randn(F_in, F_out)
a = torch.randn(2 * F_out)
out1 = {fn}(A, X, W, a, negative_slope=0.01)
out2 = {fn}(A, X, W, a, negative_slope=0.5)
assert not torch.allclose(out1, out2, atol=1e-6), 'Different negative_slope should produce different outputs'
""",
        },
    ],
    "solution": '''def gat_layer(A, X, W, a, negative_slope=0.2):
    H = X @ W
    F_out = H.size(-1)
    e = (H @ a[:F_out]).unsqueeze(1) + (H @ a[F_out:]).unsqueeze(0)
    e = torch.nn.functional.leaky_relu(e, negative_slope)
    e = e.masked_fill(A == 0, float('-inf'))
    alpha = torch.softmax(e, dim=-1)
    return alpha @ H''',
}
