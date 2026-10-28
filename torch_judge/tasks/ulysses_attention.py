"""Ulysses sequence parallelism task."""

TASK = {
    "title": "Ulysses Sequence Parallelism",
    "title_zh": "Ulysses 序列并行",
    "difficulty": "Hard",
    "description_en": "Implement DeepSpeed-Ulysses sequence parallelism for attention, simulated on one process with lists of tensors.\n\nThe activations are sharded along the sequence across P ranks: rank r holds positions `[r*S/P, (r+1)*S/P)` for all heads. Ulysses re-shards with two all-to-all collectives. The first turns sequence-parallel into head-parallel, so each rank holds the full sequence for H/P heads and runs ordinary attention. The second turns the output back into sequence-parallel. The two all-to-alls are this scheme's communication cost, and each moves roughly (P-1)/P of a rank's shard. Ring attention takes a different route: it keeps heads whole and rotates K/V shards around a ring of P point-to-point exchanges, which can be overlapped with compute. Ulysses needs no ring loop, but it requires `H` to be divisible by `P`.\n\n**Signature:** `ulysses_attention(q_shards, k_shards, v_shards, causal=False) -> list[Tensor]`\n\n**Parameters:**\n- `q_shards, k_shards, v_shards` — lists of `P` tensors, each `(B, S/P, H, D)`; rank `r` holds sequence positions `[r*S/P, (r+1)*S/P)`. `H` is divisible by `P`, and `S/P` is an integer.\n- `causal` — if True, position `i` attends only to positions `j <= i` of the full sequence (global positions, not per shard)\n\n**Returns:** a list of `P` tensors, each `(B, S/P, H, D)`. Entry `r` is rank `r`'s output for its own sequence positions, with all heads.\n\n**Algorithm:**\n```\n# all-to-all #1: sequence-parallel -> head-parallel\nrank r gathers head block [r*H/P, (r+1)*H/P) from every rank s\n  and concatenates them along the sequence -> (B, S, H/P, D)\n# local attention on the full sequence for H/P heads\nout_r = softmax(q_r k_r^T / sqrt(D), causal mask if causal) @ v_r   # (B, S, H/P, D)\n# all-to-all #2: head-parallel -> sequence-parallel\nrank s takes sequence chunk [s*S/P, (s+1)*S/P) from every out_r\n  and concatenates them along the heads -> (B, S/P, H, D)\n```\n\n**Constraints:**\n- The scale is `1 / sqrt(D)`.\n- The output must have all `H` heads in their original order, so head block `r` (not some other block) must be the one rank `r` computes in step 1.\n- The input lists are not modified.\n- Gradients must flow to every shard in all three lists.",
    "description_zh": "实现 DeepSpeed-Ulysses 序列并行的注意力（在单进程中用张量列表模拟）。\n\n激活值沿序列维切分到 P 个 rank 上：rank `r` 持有全部头在位置 `[r*S/P, (r+1)*S/P)` 上的数据。Ulysses 通过两次 all-to-all 集合通信重新分片。第一次把序列并行转为头并行，使每个 rank 持有完整序列的 H/P 个头，并执行普通注意力；第二次把输出转回序列并行。这两次 all-to-all 就是本方案的通信开销，每次大约搬运每个 rank 分片的 (P-1)/P。环形注意力（Ring Attention）则是另一条路线：它保持所有头完整，把 K/V 分片沿环做 P 次点对点交换，并且可以与计算重叠。Ulysses 不需要环形循环，但要求 `H` 能被 `P` 整除。\n\n**签名:** `ulysses_attention(q_shards, k_shards, v_shards, causal=False) -> list[Tensor]`\n\n**参数:**\n- `q_shards, k_shards, v_shards` — 各为 `P` 个张量组成的列表，每个形状 `(B, S/P, H, D)`；rank `r` 持有序列位置 `[r*S/P, (r+1)*S/P)`。`H` 能被 `P` 整除，且 `S/P` 为整数。\n- `causal` — 若为 True，位置 `i` 只关注完整序列中满足 `j <= i` 的位置 `j`（使用全局位置，而不是每个分片内部的位置）\n\n**返回:** 长度为 `P` 的列表，每个元素形状 `(B, S/P, H, D)`。第 `r` 项是 rank `r` 在自身序列位置上、包含全部头的输出。\n\n**算法:**\n```\n# all-to-all #1：序列并行 -> 头并行\nrank r 从每个 rank s 收集头块 [r*H/P, (r+1)*H/P)，\n  并沿序列维拼接 -> (B, S, H/P, D)\n# 对 H/P 个头在完整序列上做本地注意力\nout_r = softmax(q_r k_r^T / sqrt(D), 若 causal 则加因果掩码) @ v_r   # (B, S, H/P, D)\n# all-to-all #2：头并行 -> 序列并行\nrank s 从每个 out_r 取序列块 [s*S/P, (s+1)*S/P)，\n  并沿头维拼接 -> (B, S/P, H, D)\n```\n\n**约束:**\n- 缩放因子为 `1 / sqrt(D)`。\n- 输出必须包含全部 `H` 个头且顺序与原始一致，因此 rank `r` 在第一步计算的必须是头块 `r`。\n- 输入列表不得被修改。\n- 梯度需要回传到三个列表中的每一个分片。",
    "function_name": "ulysses_attention",
    "hint": "all-to-all #1: rank r takes head block r from every shard, cat along seq -> (B, S, H/P, D)\nfor each rank r: out_r = attention(q_r, k_r, v_r) over the full S (causal mask if causal)\nall-to-all #2: rank s takes seq chunk s from every out_r, cat along heads -> (B, S/P, H, D)",
    "hint_zh": "all-to-all #1：rank r 从每个分片取头块 r，沿序列维拼接 -> (B, S, H/P, D)\n对每个 rank r：out_r = attention(q_r, k_r, v_r)，在完整 S 上计算（若 causal 则加因果掩码）\nall-to-all #2：rank s 从每个 out_r 取序列块 s，沿头维拼接 -> (B, S/P, H, D)",
    "tests": [
        {
            "name": "Output count and shapes",
            "code": """
import torch
torch.manual_seed(0)
B, S, H, D, P = 2, 8, 4, 8, 4
q_shards = [torch.randn(B, S // P, H, D) for _ in range(P)]
k_shards = [torch.randn(B, S // P, H, D) for _ in range(P)]
v_shards = [torch.randn(B, S // P, H, D) for _ in range(P)]
outs = {fn}(q_shards, k_shards, v_shards, causal=False)
assert isinstance(outs, (list, tuple)) and len(outs) == P, f'Expected {P} output shards, got {len(outs) if isinstance(outs, (list, tuple)) else type(outs)}'
for r, o in enumerate(outs):
    assert o.shape == (B, S // P, H, D), f'shard {r}: expected {(B, S // P, H, D)}, got {tuple(o.shape)}'
""",
        },
        {
            "name": "Non-causal matches full attention for P in {1, 2, 4}",
            "code": """
import torch
torch.manual_seed(1)

def ref_full(q, k, v, causal):
    B, S, H, D = q.shape
    sc = torch.einsum('bihd,bjhd->bhij', q, k) * D ** -0.5
    if causal:
        m = torch.tril(torch.ones(S, S, dtype=torch.bool))
        sc = sc.masked_fill(~m, float('-inf'))
    return torch.einsum('bhij,bjhd->bihd', torch.softmax(sc, dim=-1), v)

B, S, H, D = 2, 8, 4, 6
for P in [1, 2, 4]:
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    outs = {fn}(list(q.split(S // P, dim=1)), list(k.split(S // P, dim=1)), list(v.split(S // P, dim=1)), causal=False)
    got = torch.cat(outs, dim=1)
    ref = ref_full(q, k, v, causal=False)
    diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=1e-5), f'P={P} non-causal: max diff {diff:.2e}'
""",
        },
        {
            "name": "Causal matches full causal attention for P in {1, 2, 4}",
            "code": """
import torch
torch.manual_seed(2)

def ref_full(q, k, v, causal):
    B, S, H, D = q.shape
    sc = torch.einsum('bihd,bjhd->bhij', q, k) * D ** -0.5
    if causal:
        m = torch.tril(torch.ones(S, S, dtype=torch.bool))
        sc = sc.masked_fill(~m, float('-inf'))
    return torch.einsum('bhij,bjhd->bihd', torch.softmax(sc, dim=-1), v)

B, S, H, D = 2, 8, 4, 6
for P in [1, 2, 4]:
    q = torch.randn(B, S, H, D)
    k = torch.randn(B, S, H, D)
    v = torch.randn(B, S, H, D)
    outs = {fn}(list(q.split(S // P, dim=1)), list(k.split(S // P, dim=1)), list(v.split(S // P, dim=1)), causal=True)
    got = torch.cat(outs, dim=1)
    ref = ref_full(q, k, v, causal=True)
    diff = (got - ref).abs().max().item()
    assert torch.allclose(got, ref, atol=1e-5), f'P={P} causal: max diff {diff:.2e}'
""",
        },
        {
            "name": "Causal first token attends only to itself",
            "code": """
import torch
torch.manual_seed(3)
B, S, H, D, P = 1, 4, 2, 3, 2
q = torch.randn(B, S, H, D)
k = torch.randn(B, S, H, D)
v = torch.randn(B, S, H, D)
outs = {fn}(list(q.split(S // P, dim=1)), list(k.split(S // P, dim=1)), list(v.split(S // P, dim=1)), causal=True)
# Rank 0 holds positions 0 and 1; position 0 sees only itself, so its output is v[:, 0]
assert torch.allclose(outs[0][:, 0], v[:, 0], atol=1e-6), 'causal position 0 should output its own value'
# Rank 1 holds positions 2 and 3; position 2 must not see position 3 (global causal positions)
sc = torch.einsum('bhd,bjhd->bhj', q[:, 2], k[:, :3]) * D ** -0.5
ref2 = torch.einsum('bhj,bjhd->bhd', torch.softmax(sc, dim=-1), v[:, :3])
diff = (outs[1][:, 0] - ref2).abs().max().item()
assert torch.allclose(outs[1][:, 0], ref2, atol=1e-5), f'causal position 2 uses wrong keys (max diff {diff:.2e})'
""",
        },
        {
            "name": "Heads that differ strongly map to the right ranks",
            "code": """
import torch
torch.manual_seed(4)

def ref_full(q, k, v, causal):
    B, S, H, D = q.shape
    sc = torch.einsum('bihd,bjhd->bhij', q, k) * D ** -0.5
    if causal:
        m = torch.tril(torch.ones(S, S, dtype=torch.bool))
        sc = sc.masked_fill(~m, float('-inf'))
    return torch.einsum('bhij,bjhd->bihd', torch.softmax(sc, dim=-1), v)

B, S, H, D = 1, 8, 4, 4
# Head h has values centred at 100 * h, so any head permutation changes the output a lot
offset = 100.0 * torch.arange(H, dtype=torch.float32).view(1, 1, H, 1)
for P in [2, 4]:
    for causal in [False, True]:
        q = torch.randn(B, S, H, D)
        k = torch.randn(B, S, H, D)
        v = torch.randn(B, S, H, D) + offset
        outs = {fn}(list(q.split(S // P, dim=1)), list(k.split(S // P, dim=1)), list(v.split(S // P, dim=1)), causal=causal)
        got = torch.cat(outs, dim=1)
        ref = ref_full(q, k, v, causal)
        diff = (got - ref).abs().max().item()
        assert torch.allclose(got, ref, atol=1e-4), f'P={P} causal={causal}: head-to-rank mapping is wrong (max diff {diff:.2e})'
""",
        },
        {
            "name": "Inputs are not mutated",
            "code": """
import torch
torch.manual_seed(5)
B, S, H, D, P = 2, 8, 4, 4, 2
q = torch.randn(B, S, H, D)
k = torch.randn(B, S, H, D)
v = torch.randn(B, S, H, D)
for causal in [False, True]:
    q_sh = [x.clone() for x in q.split(S // P, dim=1)]
    k_sh = [x.clone() for x in k.split(S // P, dim=1)]
    v_sh = [x.clone() for x in v.split(S // P, dim=1)]
    snap = [x.clone() for x in q_sh + k_sh + v_sh]
    {fn}(q_sh, k_sh, v_sh, causal=causal)
    for i, (before, after) in enumerate(zip(snap, q_sh + k_sh + v_sh)):
        assert torch.equal(before, after), f'causal={causal}: input shard {i} was modified in place'
""",
        },
        {
            "name": "Gradients flow to all shards",
            "code": """
import torch
torch.manual_seed(6)
B, S, H, D, P = 1, 8, 4, 4, 4
Sl = S // P
qs = [torch.randn(B, Sl, H, D, requires_grad=True) for _ in range(P)]
ks = [torch.randn(B, Sl, H, D, requires_grad=True) for _ in range(P)]
vs = [torch.randn(B, Sl, H, D, requires_grad=True) for _ in range(P)]
W = [torch.randn(B, Sl, H, D) for _ in range(P)]
outs = {fn}(qs, ks, vs, causal=True)
loss = sum((o * w).sum() for o, w in zip(outs, W))
loss.backward()

# Reference gradients from a full-sequence computation
Q = torch.cat([x.detach() for x in qs], dim=1).requires_grad_(True)
K = torch.cat([x.detach() for x in ks], dim=1).requires_grad_(True)
V = torch.cat([x.detach() for x in vs], dim=1).requires_grad_(True)
sc = torch.einsum('bihd,bjhd->bhij', Q, K) * D ** -0.5
m = torch.tril(torch.ones(S, S, dtype=torch.bool))
sc = sc.masked_fill(~m, float('-inf'))
ref = torch.einsum('bhij,bjhd->bihd', torch.softmax(sc, dim=-1), V)
(ref * torch.cat(W, dim=1)).sum().backward()

for name, shards, full in [('q', qs, Q), ('k', ks, K), ('v', vs, V)]:
    for r, x in enumerate(shards):
        assert x.grad is not None, f'no gradient reached {name} shard {r}'
        ref_grad = full.grad[:, r * Sl:(r + 1) * Sl]
        assert torch.allclose(x.grad, ref_grad, atol=1e-5), f'gradient for {name} shard {r} does not match reference'
""",
        },
    ],
    "solution": '''def ulysses_attention(q_shards, k_shards, v_shards, causal=False):
    P = len(q_shards)
    B, S_local, H, D = q_shards[0].shape
    H_local = H // P
    S = S_local * P
    scale = D ** -0.5

    def all_to_all_seq_to_head(shards):
        # Rank r takes head block r from every sequence shard, then joins them along the sequence
        return [
            torch.cat([shards[s][:, :, r * H_local:(r + 1) * H_local] for s in range(P)], dim=1)
            for r in range(P)
        ]  # P tensors of (B, S, H/P, D)

    q_full = all_to_all_seq_to_head(q_shards)
    k_full = all_to_all_seq_to_head(k_shards)
    v_full = all_to_all_seq_to_head(v_shards)

    # Causal mask over global positions of the full sequence
    mask = torch.ones(S, S, dtype=torch.bool, device=q_shards[0].device)
    if causal:
        mask = torch.tril(mask)

    # Each rank runs ordinary attention over the full sequence for its H/P heads
    outs = []
    for q, k, v in zip(q_full, k_full, v_full):
        scores = torch.einsum('bihd,bjhd->bhij', q, k) * scale  # (B, H/P, S, S)
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum('bhij,bjhd->bihd', attn, v))   # (B, S, H/P, D)

    # All-to-all back: rank s keeps sequence chunk s from every head block, joined along the heads
    result = []
    for s in range(P):
        chunks = [outs[r][:, s * S_local:(s + 1) * S_local] for r in range(P)]
        result.append(torch.cat(chunks, dim=2))  # (B, S/P, H, D)
    return result''',
    "demo": """torch.manual_seed(0)
B, S, H, D, P = 1, 8, 4, 4, 4
q = torch.randn(B, S, H, D)
k = torch.randn(B, S, H, D)
v = torch.randn(B, S, H, D)
Sl = S // P

outs = ulysses_attention(list(q.split(Sl, dim=1)), list(k.split(Sl, dim=1)), list(v.split(Sl, dim=1)), causal=True)
print(f'ranks={P}  shard shape={tuple(outs[0].shape)}  heads per rank in step 1={H // P}')

sc = torch.einsum('bihd,bjhd->bhij', q, k) * D ** -0.5
sc = sc.masked_fill(~torch.tril(torch.ones(S, S, dtype=torch.bool)), float('-inf'))
ref = torch.einsum('bhij,bjhd->bihd', torch.softmax(sc, dim=-1), v)
for r, o in enumerate(outs):
    diff = (o - ref[:, r * Sl:(r + 1) * Sl]).abs().max().item()
    print(f'rank {r}: positions [{r * Sl}, {(r + 1) * Sl})  max_diff={diff:.2e}')""",
}
