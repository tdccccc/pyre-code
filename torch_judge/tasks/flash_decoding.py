"""Flash-Decoding (split-KV attention) task."""

TASK = {
    "title": "Flash-Decoding (Split-KV)",
    "title_zh": "Flash-Decoding（KV 分块）",
    "difficulty": "Hard",
    "description_en": "Implement Flash-Decoding, split-KV attention for single-token decoding.\n\nDuring decode, one query per head attends to a very long KV cache, so there is little parallelism over queries. Flash-Decoding splits the KV sequence into chunks that are processed in parallel. Each chunk yields a partial output and its log-sum-exp (lse), and the partials are merged exactly, so the result equals full softmax attention.\n\n**Signature:** `flash_decoding(q, K, V, num_splits) -> tuple[Tensor, Tensor]`\n\n**Parameters:**\n- `q` — query tensor `(B, H, D)`, one query per head (single decode step)\n- `K, V` — key and value tensors `(B, H, S, D)`\n- `num_splits` — integer with `1 <= num_splits <= S`, the number of KV chunks\n\n**Returns:** a tuple `(out, lse)`:\n- `out` — `(B, H, D)`, the attention output\n- `lse` — `(B, H)`, the logsumexp over all `S` scaled scores\n\n**Algorithm:**\n```\nchunks = torch.tensor_split(K, num_splits, dim=2)   # same split for V\nfor each chunk c (in order):\n  s_c = q·K_c / sqrt(D)            # (B, H, len_c)\n  lse_c = logsumexp(s_c, dim=-1)   # (B, H)\n  out_c = softmax(s_c) @ V_c       # softmax over this chunk only\nlse = logsumexp over c of lse_c\nout = sum over c of exp(lse_c - lse) * out_c\n```\n\n**Constraints:**\n- The scale is `1 / sqrt(D)`.\n- Chunk sizes follow `torch.tensor_split` exactly: sizes may be unequal, and the first `S % num_splits` chunks are one token longer than the rest.\n- Do not build any tensor of shape `(B, H, S)` or `(B, H, 1, S)`. The grader records the shape of every tensor returned by a torch operation during the call.\n- Use a numerically stable log-sum-exp (subtract the maximum). Large-magnitude `q` must not produce `nan` or `inf`.\n- Gradients must flow to `q`, `K` and `V`.",
    "description_zh": "实现 Flash-Decoding（分块 KV 注意力，用于单 token 解码）。\n\n解码阶段每个头的单个查询要关注很长的 KV 缓存，沿查询维度的并行度很低。Flash-Decoding 把 KV 序列切成多个分块并行处理。每个分块给出局部输出及其 log-sum-exp（lse），再把局部结果精确合并，结果与完整的 softmax 注意力相同。\n\n**签名:** `flash_decoding(q, K, V, num_splits) -> tuple[Tensor, Tensor]`\n\n**参数:**\n- `q` — 查询张量 `(B, H, D)`，每个头一个查询（单步解码）\n- `K, V` — 键与值张量 `(B, H, S, D)`\n- `num_splits` — 整数，满足 `1 <= num_splits <= S`，即 KV 分块数\n\n**返回:** 元组 `(out, lse)`:\n- `out` — `(B, H, D)`，注意力输出\n- `lse` — `(B, H)`，对全部 `S` 个缩放分数取 `logsumexp` 的结果\n\n**算法:**\n```\nchunks = torch.tensor_split(K, num_splits, dim=2)   # V 使用相同的切分\n对每个分块 c（按顺序）:\n  s_c = q·K_c / sqrt(D)            # (B, H, len_c)\n  lse_c = logsumexp(s_c, dim=-1)   # (B, H)\n  out_c = softmax(s_c) @ V_c       # 仅在该分块内做 softmax\nlse = 对所有 c 的 lse_c 取 logsumexp\nout = 对所有 c 求 exp(lse_c - lse) * out_c 之和\n```\n\n**约束:**\n- 缩放因子为 `1 / sqrt(D)`。\n- 分块大小严格遵循 `torch.tensor_split`：各块大小可以不同，前 `S % num_splits` 块比其余块多一个 token。\n- 不得构造形状为 `(B, H, S)` 或 `(B, H, 1, S)` 的张量。测试会记录调用过程中每个 torch 运算返回张量的形状。\n- 使用数值稳定的 log-sum-exp（减去最大值）。即使 `q` 的量级很大，也不能产生 `nan` 或 `inf`。\n- 梯度需要回传到 `q`、`K` 和 `V`。",
    "function_name": "flash_decoding",
    "hint": "chunks = tensor_split(K, num_splits, dim=2); same for V\nfor each chunk K_c, V_c:\n  s = einsum('bhd,bhsd->bhs', q, K_c) * scale\n  lse_c = logsumexp(s, -1);  out_c = exp(s - lse_c) @ V_c\nlse = logsumexp(stack(lse_c))\nout = sum_c exp(lse_c - lse) * out_c",
    "hint_zh": "chunks = tensor_split(K, num_splits, dim=2)，V 同理\n对每个分块 K_c, V_c:\n  s = einsum('bhd,bhsd->bhs', q, K_c) * scale\n  lse_c = logsumexp(s, -1);  out_c = exp(s - lse_c) @ V_c\nlse = logsumexp(stack(lse_c))\nout = sum_c exp(lse_c - lse) * out_c",
    "tests": [
        {
            "name": "Output shapes",
            "code": """
import torch
torch.manual_seed(0)
B, H, S, D = 2, 3, 10, 4
q = torch.randn(B, H, D)
K = torch.randn(B, H, S, D)
V = torch.randn(B, H, S, D)
out, lse = {fn}(q, K, V, num_splits=3)
assert out.shape == (B, H, D), f'Expected out shape {(B, H, D)}, got {tuple(out.shape)}'
assert lse.shape == (B, H), f'Expected lse shape {(B, H)}, got {tuple(lse.shape)}'
""",
        },
        {
            "name": "Hand-computed two-token case",
            "code": """
import torch
import math
q = torch.tensor([[[1.0]]])                      # (1, 1, 1); D = 1 so scale = 1
K = torch.tensor([[[[0.0], [math.log(3.0)]]]])   # (1, 1, 2, 1)
V = torch.tensor([[[[2.0], [6.0]]]])             # (1, 1, 2, 1)
out, lse = {fn}(q, K, V, num_splits=2)
# scores = (0, ln 3) -> softmax = (1/4, 3/4) -> out = 0.5 + 4.5 = 5, lse = ln(1 + 3) = ln 4
assert torch.allclose(out, torch.tensor([[[5.0]]]), atol=1e-5), f'Expected out 5.0, got {out.flatten().tolist()}'
assert torch.allclose(lse, torch.tensor([[math.log(4.0)]]), atol=1e-5), f'Expected lse ln(4), got {lse.flatten().tolist()}'
""",
        },
        {
            "name": "Matches standard attention for many num_splits",
            "code": """
import torch
torch.manual_seed(1)
B, H, S, D = 2, 3, 37, 8
q = torch.randn(B, H, D)
K = torch.randn(B, H, S, D)
V = torch.randn(B, H, S, D)
scores = torch.einsum('bhd,bhsd->bhs', q, K) * D ** -0.5
ref_out = torch.einsum('bhs,bhsd->bhd', torch.softmax(scores, dim=-1), V)
for n in [1, 2, 3, 4, 7, S]:
    out, _ = {fn}(q, K, V, num_splits=n)
    diff = (out - ref_out).abs().max().item()
    assert torch.allclose(out, ref_out, atol=1e-5), f'num_splits={n}: output differs from standard attention (max diff {diff:.2e})'
""",
        },
        {
            "name": "Uneven splits match standard attention",
            "code": """
import torch
torch.manual_seed(2)
B, H, S, D = 1, 2, 10, 4
q = torch.randn(B, H, D)
K = torch.randn(B, H, S, D)
V = torch.randn(B, H, S, D)
scores = torch.einsum('bhd,bhsd->bhs', q, K) * D ** -0.5
ref_out = torch.einsum('bhs,bhsd->bhd', torch.softmax(scores, dim=-1), V)
# S = 10 split 3 -> (4, 3, 3); split 4 -> (3, 3, 2, 2); split 6 -> (2, 2, 2, 2, 1, 1)
for n in [3, 4, 6]:
    out, _ = {fn}(q, K, V, num_splits=n)
    diff = (out - ref_out).abs().max().item()
    assert torch.allclose(out, ref_out, atol=1e-5), f'num_splits={n} (uneven): max diff {diff:.2e}'
""",
        },
        {
            "name": "lse equals logsumexp of scaled scores",
            "code": """
import torch
torch.manual_seed(3)
B, H, S, D = 2, 3, 23, 16
q = torch.randn(B, H, D)
K = torch.randn(B, H, S, D)
V = torch.randn(B, H, S, D)
scores = torch.einsum('bhd,bhsd->bhs', q, K) * D ** -0.5
ref_lse = torch.logsumexp(scores, dim=-1)
for n in [1, 3, 5]:
    _, lse = {fn}(q, K, V, num_splits=n)
    assert lse.shape == (B, H), f'num_splits={n}: lse shape {tuple(lse.shape)}'
    diff = (lse - ref_lse).abs().max().item()
    assert torch.allclose(lse, ref_lse, atol=1e-5), f'num_splits={n}: lse differs from logsumexp (max diff {diff:.2e})'
""",
        },
        {
            "name": "Numerically stable with large-magnitude q",
            "code": """
import torch
torch.manual_seed(4)
B, H, S, D = 2, 2, 37, 8
q = torch.randn(B, H, D) * 50
K = torch.randn(B, H, S, D)
V = torch.randn(B, H, S, D)
out, lse = {fn}(q, K, V, num_splits=4)
assert torch.isfinite(out).all(), 'output contains nan/inf for large-magnitude q'
assert torch.isfinite(lse).all(), 'lse contains nan/inf for large-magnitude q'
# Reference in float64
q64, K64, V64 = q.double(), K.double(), V.double()
s64 = torch.einsum('bhd,bhsd->bhs', q64, K64) * D ** -0.5
ref_out = torch.einsum('bhs,bhsd->bhd', torch.softmax(s64, dim=-1), V64)
ref_lse = torch.logsumexp(s64, dim=-1)
assert torch.allclose(out.double(), ref_out, atol=1e-4), f'large-q output mismatch (max diff {(out.double() - ref_out).abs().max().item():.2e})'
assert torch.allclose(lse.double(), ref_lse, atol=1e-3), f'large-q lse mismatch (max diff {(lse.double() - ref_lse).abs().max().item():.2e})'
""",
        },
        {
            "name": "Never materializes full (B, H, S) scores",
            "code": """
import torch
torch.manual_seed(5)
B, H, S, D = 2, 3, 37, 8
q = torch.randn(B, H, D)
K = torch.randn(B, H, S, D)
V = torch.randn(B, H, S, D)

class ShapeRecorder(torch.overrides.TorchFunctionMode):
    def __init__(self):
        super().__init__()
        self.shapes = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        items = out if isinstance(out, (tuple, list)) else (out,)
        for item in items:
            if isinstance(item, torch.Tensor):
                self.shapes.append(tuple(item.shape))
        return out

rec = ShapeRecorder()
with rec:
    out, lse = {fn}(q, K, V, num_splits=4)
assert len(rec.shapes) > 0, 'no torch operations were recorded'
forbidden = [(B, H, S), (B, H, 1, S)]
for shape in rec.shapes:
    assert shape not in forbidden, f'full score tensor of shape {shape} was materialized (S={S})'
""",
        },
        {
            "name": "Gradients flow to q, K, V",
            "code": """
import torch
torch.manual_seed(6)
B, H, S, D = 1, 2, 9, 4
q = torch.randn(B, H, D, requires_grad=True)
K = torch.randn(B, H, S, D, requires_grad=True)
V = torch.randn(B, H, S, D, requires_grad=True)
out, lse = {fn}(q, K, V, num_splits=3)
(out.sum() + lse.sum()).backward()

q2 = q.detach().clone().requires_grad_(True)
K2 = K.detach().clone().requires_grad_(True)
V2 = V.detach().clone().requires_grad_(True)
s2 = torch.einsum('bhd,bhsd->bhs', q2, K2) * D ** -0.5
ref_out = torch.einsum('bhs,bhsd->bhd', torch.softmax(s2, dim=-1), V2)
(ref_out.sum() + torch.logsumexp(s2, dim=-1).sum()).backward()

for name, t, ref in [('q', q, q2), ('K', K, K2), ('V', V, V2)]:
    assert t.grad is not None, f'no gradient reached {name}'
    assert torch.allclose(t.grad, ref.grad, atol=1e-5), f'gradient for {name} does not match reference'
""",
        },
    ],
    "solution": '''def flash_decoding(q, K, V, num_splits):
    B, H, D = q.shape
    scale = D ** -0.5

    K_chunks = torch.tensor_split(K, num_splits, dim=2)
    V_chunks = torch.tensor_split(V, num_splits, dim=2)

    outs, lses = [], []
    for Kc, Vc in zip(K_chunks, V_chunks):
        # Scores for this chunk only: (B, H, len_c)
        scores = torch.einsum('bhd,bhsd->bhs', q, Kc) * scale
        # Partial result of this chunk: its own logsumexp and softmax-weighted V
        lse_i = torch.logsumexp(scores, dim=-1)                  # (B, H)
        p = torch.exp(scores - lse_i.unsqueeze(-1))              # softmax within the chunk
        outs.append(torch.einsum('bhs,bhsd->bhd', p, Vc))        # (B, H, D)
        lses.append(lse_i)

    outs = torch.stack(outs, dim=0)                              # (num_splits, B, H, D)
    lses = torch.stack(lses, dim=0)                              # (num_splits, B, H)

    # Merge: rescale each partial by its share of the total softmax mass
    lse = torch.logsumexp(lses, dim=0)                           # (B, H)
    weights = torch.exp(lses - lse)                              # (num_splits, B, H)
    out = (weights.unsqueeze(-1) * outs).sum(dim=0)              # (B, H, D)
    return out, lse''',
    "demo": """torch.manual_seed(0)
B, H, S, D = 2, 4, 37, 16
q = torch.randn(B, H, D)
K = torch.randn(B, H, S, D)
V = torch.randn(B, H, S, D)

scores = torch.einsum('bhd,bhsd->bhs', q, K) * D ** -0.5
ref_out = torch.einsum('bhs,bhsd->bhd', torch.softmax(scores, dim=-1), V)
ref_lse = torch.logsumexp(scores, dim=-1)

for n in [1, 2, 4, 5, 8]:
    sizes = [c.shape[2] for c in torch.tensor_split(K, n, dim=2)]
    out, lse = flash_decoding(q, K, V, n)
    out_diff = (out - ref_out).abs().max().item()
    lse_diff = (lse - ref_lse).abs().max().item()
    print(f'num_splits={n}  chunk sizes={sizes}  out_diff={out_diff:.2e}  lse_diff={lse_diff:.2e}')""",
}
