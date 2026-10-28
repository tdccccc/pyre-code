"""Packed variable-length attention task."""

TASK = {
    "title": "Packed Variable-Length Attention",
    "title_zh": "变长序列打包注意力",
    "difficulty": "Medium",
    "description_en": "Implement attention over packed variable-length sequences (flash_attn_varlen_func style).\n\nSequences of different lengths are concatenated along the token axis into one tensor with no padding, so no compute is wasted on pad tokens. The boundaries are given by `cu_seqlens`, and every token may only attend to tokens of its own sequence.\n\n**Signature:** `varlen_attention(q, k, v, cu_seqlens, causal=True) -> Tensor`\n\n**Parameters:**\n- `q, k, v` — tensors `(T, H, D)`, where $T$ is the total number of packed tokens\n- `cu_seqlens` — int64 tensor `(N+1,)` of cumulative boundaries: sequence $j$ occupies packed tokens `[cu_seqlens[j], cu_seqlens[j+1])`\n- `causal` — if True, token $i$ attends to token $j$ of the same sequence only when $j \\le i$ (packed order inside a sequence matches the position within it); if False, it attends to every token of its own sequence\n\n**Returns:** output tensor `(T, H, D)`, one row per packed token. It equals running standard attention separately on each sequence and concatenating the results.\n\n**Constraints:**\n- `cu_seqlens[0] == 0`, `cu_seqlens[-1] == T`, and the values are non-decreasing. Zero-length sequences (equal consecutive boundaries) are allowed and contribute no tokens.\n- The scale is $1/\\sqrt{D}$, applied per head; heads are independent.\n- A token must never attend to a token of a different sequence, even though the sequences sit side by side in one tensor.",
    "description_zh": "实现打包变长序列的注意力（flash_attn_varlen_func 风格）。\n\n不同长度的序列沿 token 维拼接成一个张量，不做 padding，因此不会在填充 token 上浪费计算。序列边界由 `cu_seqlens` 给出，每个 token 只能关注同一序列内的 token。\n\n**签名:** `varlen_attention(q, k, v, cu_seqlens, causal=True) -> Tensor`\n\n**参数:**\n- `q, k, v` — 形状 `(T, H, D)` 的张量，$T$ 为打包后的 token 总数\n- `cu_seqlens` — int64 张量，形状 `(N+1,)`，为累积边界：第 $j$ 个序列占据打包后的 token `[cu_seqlens[j], cu_seqlens[j+1])`\n- `causal` — 若为 True，同一序列内 token $i$ 只关注满足 $j \\le i$ 的 token $j$（序列内的打包顺序与其内部位置一致）；若为 False，则关注同一序列的全部 token\n\n**返回:** 形状 `(T, H, D)` 的输出张量，每个打包 token 对应一行。其结果等于对每个序列分别执行标准注意力，再将结果拼接。\n\n**约束:**\n- `cu_seqlens[0] == 0`，`cu_seqlens[-1] == T`，且取值非递减。允许长度为零的序列（相邻边界相等），它们不包含任何 token。\n- 缩放因子为 $1/\\sqrt{D}$，按头独立应用；各头相互独立。\n- 即使不同序列在同一张量中相邻排列，token 也绝不能关注其他序列的 token。",
    "function_name": "varlen_attention",
    "hint": "seg[t] = sequence id of token t = searchsorted(cu_seqlens, t, right=True) - 1\nmask[i, j] = (seg[i] == seg[j])          # block-diagonal: same sequence only\nif causal: mask &= (j <= i)\nscores = einsum('ihd,jhd->hij', q, k) * scale; scores[~mask] = -inf\nout = einsum('hij,jhd->ihd', softmax(scores, -1), v)",
    "hint_zh": "seg[t] = token t 所属的序列编号 = searchsorted(cu_seqlens, t, right=True) - 1\nmask[i, j] = (seg[i] == seg[j])          # 块对角：只允许同一序列\n若 causal：mask &= (j <= i)\nscores = einsum('ihd,jhd->hij', q, k) * scale; scores[~mask] = -inf\nout = einsum('hij,jhd->ihd', softmax(scores, -1), v)",
    "tests": [
        {
            "name": "Output shape",
            "code": """
import torch
torch.manual_seed(0)
T, H, D = 7, 2, 4
q = torch.randn(T, H, D)
k = torch.randn(T, H, D)
v = torch.randn(T, H, D)
cu = torch.tensor([0, 3, 5, 7], dtype=torch.int64)
out = {fn}(q, k, v, cu)
assert out.shape == (T, H, D), f'Expected ({T}, {H}, {D}), got {tuple(out.shape)}'
assert out.dtype == q.dtype, f'Expected dtype {q.dtype}, got {out.dtype}'
""",
        },
        {
            "name": "Single sequence equals plain causal attention",
            "code": """
import torch
torch.manual_seed(1)
T, H, D = 6, 2, 4
q = torch.randn(T, H, D)
k = torch.randn(T, H, D)
v = torch.randn(T, H, D)
cu = torch.tensor([0, T], dtype=torch.int64)
out = {fn}(q, k, v, cu, causal=True)
# Reference: standard causal attention on the one sequence
scores = torch.einsum('ihd,jhd->hij', q, k) * D ** -0.5
mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
scores = scores.masked_fill(~mask, float('-inf'))
ref = torch.einsum('hij,jhd->ihd', torch.softmax(scores, dim=-1), v)
diff = (out - ref).abs().max().item()
assert torch.allclose(out, ref, atol=1e-5), f'Single causal sequence differs from plain attention (max diff {diff:.2e})'
""",
        },
        {
            "name": "Multiple sequences (causal) match per-sequence attention",
            "code": """
import torch
torch.manual_seed(2)

def ref_per_seq(q, k, v, cu, causal):
    outs = []
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        if e == s:
            continue
        qs, ks, vs = q[s:e], k[s:e], v[s:e]
        L, H, D = qs.shape
        sc = torch.einsum('ihd,jhd->hij', qs, ks) * D ** -0.5
        if causal:
            m = torch.tril(torch.ones(L, L, dtype=torch.bool))
            sc = sc.masked_fill(~m, float('-inf'))
        outs.append(torch.einsum('hij,jhd->ihd', torch.softmax(sc, dim=-1), vs))
    return torch.cat(outs, dim=0)

T, H, D = 9, 2, 4
q = torch.randn(T, H, D)
k = torch.randn(T, H, D)
v = torch.randn(T, H, D)
cu = torch.tensor([0, 3, 5, 9], dtype=torch.int64)
out = {fn}(q, k, v, cu, causal=True)
ref = ref_per_seq(q, k, v, cu, True)
diff = (out - ref).abs().max().item()
assert torch.allclose(out, ref, atol=1e-5), f'Causal packed output differs from per-sequence attention (max diff {diff:.2e})'
""",
        },
        {
            "name": "Multiple sequences (non-causal) match per-sequence attention",
            "code": """
import torch
torch.manual_seed(3)

def ref_per_seq(q, k, v, cu, causal):
    outs = []
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        if e == s:
            continue
        qs, ks, vs = q[s:e], k[s:e], v[s:e]
        L, H, D = qs.shape
        sc = torch.einsum('ihd,jhd->hij', qs, ks) * D ** -0.5
        outs.append(torch.einsum('hij,jhd->ihd', torch.softmax(sc, dim=-1), vs))
    return torch.cat(outs, dim=0)

T, H, D = 9, 2, 4
q = torch.randn(T, H, D)
k = torch.randn(T, H, D)
v = torch.randn(T, H, D)
cu = torch.tensor([0, 3, 5, 9], dtype=torch.int64)
out = {fn}(q, k, v, cu, causal=False)
ref = ref_per_seq(q, k, v, cu, False)
diff = (out - ref).abs().max().item()
assert torch.allclose(out, ref, atol=1e-5), f'Non-causal packed output differs from per-sequence attention (max diff {diff:.2e})'
""",
        },
        {
            "name": "Hand-computed case with uniform attention",
            "code": """
import torch
torch.manual_seed(4)
T, H, D = 3, 1, 1
q = torch.zeros(T, H, D)   # all scores are 0, so attention is uniform over the allowed tokens
k = torch.randn(T, H, D)
v = torch.tensor([[[1.0]], [[2.0]], [[4.0]]])
cu = torch.tensor([0, 1, 3], dtype=torch.int64)  # sequences: [token 0] and [tokens 1, 2]
out = {fn}(q, k, v, cu, causal=True)
expected = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])  # token 2 averages v1 and v2: (2 + 4) / 2
assert torch.allclose(out, expected, atol=1e-6), f'Causal: expected {expected.flatten().tolist()}, got {out.flatten().tolist()}'
out_nc = {fn}(q, k, v, cu, causal=False)
expected_nc = torch.tensor([[[1.0]], [[3.0]], [[3.0]]])  # both tokens of seq 2 average v1 and v2
assert torch.allclose(out_nc, expected_nc, atol=1e-6), f'Non-causal: expected {expected_nc.flatten().tolist()}, got {out_nc.flatten().tolist()}'
""",
        },
        {
            "name": "Sequences do not see each other",
            "code": """
import torch
torch.manual_seed(5)
T, H, D = 9, 2, 4
cu = torch.tensor([0, 4, 9], dtype=torch.int64)
q = torch.randn(T, H, D)
k = torch.randn(T, H, D)
v = torch.randn(T, H, D)
for causal in [True, False]:
    out = {fn}(q, k, v, cu, causal=causal)
    # Change everything about the second sequence (tokens 4..8)
    k2 = k.clone()
    v2 = v.clone()
    k2[4:] = torch.randn(5, H, D)
    v2[4:] = torch.randn(5, H, D)
    out2 = {fn}(q, k2, v2, cu, causal=causal)
    assert torch.allclose(out[:4], out2[:4], atol=1e-6), f'causal={causal}: sequence 1 output changed when sequence 2 changed'
    assert not torch.allclose(out[4:], out2[4:]), f'causal={causal}: sequence 2 output should change (test is not exercising anything)'
""",
        },
        {
            "name": "Zero-length segments",
            "code": """
import torch
torch.manual_seed(6)

def ref_per_seq(q, k, v, cu, causal):
    outs = []
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        if e == s:
            continue
        qs, ks, vs = q[s:e], k[s:e], v[s:e]
        L, H, D = qs.shape
        sc = torch.einsum('ihd,jhd->hij', qs, ks) * D ** -0.5
        if causal:
            m = torch.tril(torch.ones(L, L, dtype=torch.bool))
            sc = sc.masked_fill(~m, float('-inf'))
        outs.append(torch.einsum('hij,jhd->ihd', torch.softmax(sc, dim=-1), vs))
    return torch.cat(outs, dim=0)

T, H, D = 5, 2, 4
q = torch.randn(T, H, D)
k = torch.randn(T, H, D)
v = torch.randn(T, H, D)
cu = torch.tensor([0, 2, 2, 5, 5], dtype=torch.int64)  # lengths 2, 0, 3, 0
for causal in [True, False]:
    out = {fn}(q, k, v, cu, causal=causal)
    assert out.shape == (T, H, D), f'causal={causal}: expected ({T}, {H}, {D}), got {tuple(out.shape)}'
    ref = ref_per_seq(q, k, v, cu, causal)
    diff = (out - ref).abs().max().item()
    assert torch.allclose(out, ref, atol=1e-5), f'causal={causal}: zero-length segments broke the output (max diff {diff:.2e})'
""",
        },
        {
            "name": "Length-1 sequences return their own value",
            "code": """
import torch
torch.manual_seed(7)
T, H, D = 4, 2, 3
q = torch.randn(T, H, D)
k = torch.randn(T, H, D)
v = torch.randn(T, H, D)
cu = torch.arange(T + 1, dtype=torch.int64)  # four sequences of length 1
for causal in [True, False]:
    out = {fn}(q, k, v, cu, causal=causal)
    assert torch.allclose(out, v, atol=1e-6), f'causal={causal}: a length-1 sequence should output its own v'
""",
        },
        {
            "name": "Gradients flow to q, k, v",
            "code": """
import torch
torch.manual_seed(8)

def ref_per_seq(q, k, v, cu, causal):
    outs = []
    for s, e in zip(cu[:-1].tolist(), cu[1:].tolist()):
        if e == s:
            continue
        qs, ks, vs = q[s:e], k[s:e], v[s:e]
        L, H, D = qs.shape
        sc = torch.einsum('ihd,jhd->hij', qs, ks) * D ** -0.5
        if causal:
            m = torch.tril(torch.ones(L, L, dtype=torch.bool))
            sc = sc.masked_fill(~m, float('-inf'))
        outs.append(torch.einsum('hij,jhd->ihd', torch.softmax(sc, dim=-1), vs))
    return torch.cat(outs, dim=0)

T, H, D = 7, 2, 4
cu = torch.tensor([0, 3, 7], dtype=torch.int64)
q = torch.randn(T, H, D, requires_grad=True)
k = torch.randn(T, H, D, requires_grad=True)
v = torch.randn(T, H, D, requires_grad=True)
W = torch.randn(T, H, D)
out = {fn}(q, k, v, cu, causal=True)
(out * W).sum().backward()

q2 = q.detach().clone().requires_grad_(True)
k2 = k.detach().clone().requires_grad_(True)
v2 = v.detach().clone().requires_grad_(True)
(ref_per_seq(q2, k2, v2, cu, True) * W).sum().backward()
for name, t, ref in [('q', q, q2), ('k', k, k2), ('v', v, v2)]:
    assert t.grad is not None, f'no gradient reached {name}'
    assert torch.allclose(t.grad, ref.grad, atol=1e-5), f'gradient for {name} does not match per-sequence reference'
""",
        },
    ],
    "solution": '''def varlen_attention(q, k, v, cu_seqlens, causal=True):
    T, H, D = q.shape
    scale = D ** -0.5

    pos = torch.arange(T, device=q.device)
    # Sequence id of every packed token: cu_seqlens[j] <= t < cu_seqlens[j+1]
    seg = torch.searchsorted(cu_seqlens, pos, right=True) - 1

    # Block-diagonal mask: a token may only attend within its own sequence
    mask = seg[:, None] == seg[None, :]
    if causal:
        # Packed order inside a sequence matches the position within it
        mask = mask & (pos[None, :] <= pos[:, None])

    scores = torch.einsum('ihd,jhd->hij', q, k) * scale  # (H, T, T)
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum('hij,jhd->ihd', attn, v)''',
    "demo": """torch.manual_seed(0)
T, H, D = 9, 2, 4
cu = torch.tensor([0, 3, 5, 9], dtype=torch.int64)  # lengths 3, 2, 4
q = torch.randn(T, H, D)
k = torch.randn(T, H, D)
v = torch.randn(T, H, D)

out = varlen_attention(q, k, v, cu, causal=True)
print('packed output shape:', tuple(out.shape))

# Changing sequence 3 must leave sequences 1 and 2 untouched
k2, v2 = k.clone(), v.clone()
k2[5:] += 10.0
v2[5:] -= 10.0
out2 = varlen_attention(q, k2, v2, cu, causal=True)
print('max change in sequence 1 + 2:', (out2[:5] - out[:5]).abs().max().item())
print('max change in sequence 3:', (out2[5:] - out[5:]).abs().max().item())""",
}
