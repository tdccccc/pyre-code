"""Transformer Params & FLOPs task."""

TASK = {
    "title": "Transformer Params & FLOPs",
    "title_zh": "Transformer 参数量与 FLOPs",
    "difficulty": "Medium",
    "description_en": "Count parameters and forward/training FLOPs for a decoder-only Transformer. This back-of-the-envelope estimate is the first step of every capacity plan, cluster sizing exercise and MFU target in AI infrastructure work.\n\n**Signature:** `transformer_flops(n_layers, d_model, d_ff, vocab_size, seq_len, batch_size, tied_embeddings=True) -> dict`\n\n**Parameters:**\n- `n_layers`, `d_model`, `d_ff`, `vocab_size`, `seq_len`, `batch_size` — positive integers\n- `tied_embeddings` — if True, the LM head shares its weight matrix with the token embedding\n\n**Model assumptions:** decoder-only; standard (non-gated) MLP `d_model -> d_ff -> d_model`; no biases anywhere; RoPE, so there are no learned position embeddings; every LayerNorm has gamma and beta; attention is full `seq_len x seq_len` (no causal halving).\n\n**Returns:** a dict of Python `int`s:\n- `params` — total trainable parameters\n- `forward_flops` — FLOPs for one forward pass over `tokens = batch_size * seq_len` tokens\n- `train_flops` — `3 * forward_flops` (backward costs about 2x forward)\n\n**Counting rules:**\n- One multiply-add = 2 FLOPs.\n- Count only matmuls. Ignore softmax, LayerNorm, GELU, residual adds, the 1/sqrt(d) score scaling and the embedding lookup (a table lookup is not a matmul).\n- The LM head matmul is always computed in the forward pass, even when its weight is tied to the embedding. Tying only saves parameters.\n- Attention costs QK^T and attn @ V for every (query, key) pair, so the term is `4 * batch_size * seq_len^2 * d_model` per layer.\n\n**Formula:**\n```\nparams = V*d                                   (token embedding)\n       + (V*d if not tied_embeddings)          (LM head)\n       + n_layers * (4*d^2 + 2*d*d_ff + 4*d)   (Q,K,V,O; MLP up/down; two LayerNorms)\n       + 2*d                                   (final LayerNorm)\n\nforward_flops = 2 * tokens * (n_layers * (4*d^2 + 2*d*d_ff) + V*d)     (weight matmuls, LM head included)\n              + n_layers * 4 * batch_size * seq_len^2 * d              (QK^T and attn @ V)\n\ntrain_flops   = 3 * forward_flops\n```\n\n**Worked example:** `n_layers=1, d_model=2, d_ff=4, vocab_size=3, seq_len=2, batch_size=1`, tied embeddings:\n- params = 3*2 + (4*4 + 2*2*4 + 4*2) + 2*2 = 6 + 40 + 4 = 50 (56 if untied)\n- forward_flops = 2*2*(16 + 16 + 6) + 1*4*1*2^2*2 = 152 + 32 = 184\n- train_flops = 3 * 184 = 552\n\n**Rule of thumb:** training costs about `6 * params * tokens`, since each parameter takes part in one multiply-add per token forward (2 FLOPs) and two more in backward (4 FLOPs). The rule is off here for two reasons. It ignores the attention matmuls, which scale with `seq_len` and are not part of `params`. It also counts the input embedding table, which is a lookup and costs no FLOPs, as if it were a matmul weight.",
    "description_zh": "计算 decoder-only Transformer 的参数量，以及前向与训练的 FLOPs。这种粗略估算是 AI Infra 中每次容量规划、集群规模估计和 MFU 目标设定的第一步。\n\n**签名:** `transformer_flops(n_layers, d_model, d_ff, vocab_size, seq_len, batch_size, tied_embeddings=True) -> dict`\n\n**参数:**\n- `n_layers`、`d_model`、`d_ff`、`vocab_size`、`seq_len`、`batch_size` — 正整数\n- `tied_embeddings` — 为 True 时，LM head 与 token embedding 共享同一个权重矩阵\n\n**模型假设:** decoder-only；标准（非门控）MLP `d_model -> d_ff -> d_model`；所有地方都没有偏置；使用 RoPE，因此没有可学习的位置编码；每个 LayerNorm 都有 gamma 和 beta；注意力为完整的 `seq_len x seq_len`（不做因果减半）。\n\n**返回:** 由 Python `int` 组成的字典：\n- `params` — 可训练参数总数\n- `forward_flops` — 处理 `tokens = batch_size * seq_len` 个 token 的一次前向传播 FLOPs\n- `train_flops` — `3 * forward_flops`（反向约为前向的 2 倍）\n\n**计数规则:**\n- 一次乘加 = 2 FLOPs。\n- 只统计矩阵乘。忽略 softmax、LayerNorm、GELU、残差加法、注意力分数的 1/sqrt(d) 缩放以及 embedding 查表（查表不是矩阵乘）。\n- 即使 LM head 的权重与 embedding 共享，前向中的 LM head 矩阵乘仍然会计算。权重共享只节省参数量，不节省计算量。\n- 注意力对每一对（query, key）计算 QK^T 和 attn @ V，因此每层的注意力项为 `4 * batch_size * seq_len^2 * d_model`。\n\n**公式:**\n```\nparams = V*d                                   （token embedding）\n       + (V*d 若 tied_embeddings 为 False)     （LM head）\n       + n_layers * (4*d^2 + 2*d*d_ff + 4*d)   （Q,K,V,O；MLP 上/下投影；两个 LayerNorm）\n       + 2*d                                   （最终 LayerNorm）\n\nforward_flops = 2 * tokens * (n_layers * (4*d^2 + 2*d*d_ff) + V*d)     （权重矩阵乘，包含 LM head）\n              + n_layers * 4 * batch_size * seq_len^2 * d              （QK^T 与 attn @ V）\n\ntrain_flops   = 3 * forward_flops\n```\n\n**手算示例:** `n_layers=1, d_model=2, d_ff=4, vocab_size=3, seq_len=2, batch_size=1`，权重共享：\n- params = 3*2 + (4*4 + 2*2*4 + 4*2) + 2*2 = 6 + 40 + 4 = 50（不共享则为 56）\n- forward_flops = 2*2*(16 + 16 + 6) + 1*4*1*2^2*2 = 152 + 32 = 184\n- train_flops = 3 * 184 = 552\n\n**经验法则:** 训练成本约为 `6 * params * tokens`，因为每个参数在前向中每个 token 参与一次乘加（2 FLOPs），在反向中再参与两次（共 4 FLOPs）。但这个法则在这里并不精确，原因有两点：它忽略了注意力矩阵乘，该项随 `seq_len` 增长且不计入 `params`；它还把输入 embedding 表（查表操作，不产生 FLOPs）当作矩阵乘权重计入了参数量。",
    "function_name": "transformer_flops",
    "hint": "tokens = batch_size * seq_len\nembed = V*d;  head = 0 if tied_embeddings else V*d\nper_layer = 4*d*d + 2*d*d_ff + 4*d\nparams = embed + head + n_layers*per_layer + 2*d\nforward = 2*tokens*(n_layers*(4*d*d + 2*d*d_ff) + V*d) + 4*n_layers*batch_size*seq_len**2*d\ntrain = 3*forward",
    "hint_zh": "tokens = batch_size * seq_len\nembed = V*d;  head = 0 if tied_embeddings else V*d\nper_layer = 4*d*d + 2*d*d_ff + 4*d\nparams = embed + head + n_layers*per_layer + 2*d\nforward = 2*tokens*(n_layers*(4*d*d + 2*d*d_ff) + V*d) + 4*n_layers*batch_size*seq_len**2*d\ntrain = 3*forward",
    "tests": [
        {
            "name": "Tiny hand-computed example",
            "code": """
import torch
out = {fn}(1, 2, 4, 3, 2, 1)
assert out["params"] == 50, f'Tied params: expected 50, got {out["params"]}'
assert out["forward_flops"] == 184, f'forward_flops: expected 184, got {out["forward_flops"]}'
assert out["train_flops"] == 552, f'train_flops: expected 552, got {out["train_flops"]}'
untied = {fn}(1, 2, 4, 3, 2, 1, tied_embeddings=False)
assert untied["params"] == 56, f'Untied params: expected 56, got {untied["params"]}'
assert untied["forward_flops"] == 184, f'untied forward_flops: expected 184, got {untied["forward_flops"]}'
""",
        },
        {
            "name": "Return types are int",
            "code": """
import torch
out = {fn}(2, 8, 32, 100, 4, 3)
for k in ("params", "forward_flops", "train_flops"):
    assert type(out[k]) is int, f'{k} should be int, got {type(out[k]).__name__}'
""",
        },
        {
            "name": "GPT-2 small (tied embeddings)",
            "code": """
import torch
# GPT-2 small shape: L=12, d=768, d_ff=3072, V=50257, one sequence of 1024 tokens
# params: embedding 50257*768 = 38597376
#         per layer 4*768^2 + 2*768*3072 + 4*768 = 7080960, x12 = 84971520
#         final LayerNorm 2*768 = 1536
out = {fn}(12, 768, 3072, 50257, 1024, 1)
assert out["params"] == 123570432, f'GPT-2 small params: expected 123570432, got {out["params"]}'
# forward weights: 2*1024*(12*(4*768^2 + 2*768*3072) + 50257*768) = 252993601536
# attention:       12*4*1024^2*768 = 38654705664
assert out["forward_flops"] == 291648307200, f'GPT-2 forward_flops: expected 291648307200, got {out["forward_flops"]}'
assert out["train_flops"] == 874944921600, f'GPT-2 train_flops: expected 874944921600, got {out["train_flops"]}'
""",
        },
        {
            "name": "GPT-2 small (untied) adds only the LM head weights",
            "code": """
import torch
out = {fn}(12, 768, 3072, 50257, 1024, 1, tied_embeddings=False)
assert out["params"] == 162167808, f'Untied GPT-2 params: expected 162167808, got {out["params"]}'
assert out["forward_flops"] == 291648307200, f'Untied forward_flops should equal the tied value 291648307200, got {out["forward_flops"]}'
""",
        },
        {
            "name": "Untied adds exactly V*d params; forward and train FLOPs unchanged",
            "code": """
import torch
torch.manual_seed(0)
for _ in range(4):
    L = int(torch.randint(1, 9, (1,)))
    d = 64 * int(torch.randint(1, 9, (1,)))
    dff = d * int(torch.randint(2, 6, (1,)))
    V = int(torch.randint(1000, 60000, (1,)))
    s = int(torch.randint(1, 513, (1,)))
    B = int(torch.randint(1, 17, (1,)))
    tied = {fn}(L, d, dff, V, s, B)
    untied = {fn}(L, d, dff, V, s, B, tied_embeddings=False)
    assert untied["params"] - tied["params"] == V * d, f'Untying should add V*d={V * d} params, added {untied["params"] - tied["params"]}'
    assert untied["forward_flops"] == tied["forward_flops"], 'LM head matmul runs even when tied: forward_flops must not change'
    assert untied["train_flops"] == tied["train_flops"], 'train_flops must not change when embeddings are tied'
""",
        },
        {
            "name": "Train is 3x forward; FLOPs scale linearly with batch size",
            "code": """
import torch
base = {fn}(4, 128, 512, 2000, 64, 1)
big = {fn}(4, 128, 512, 2000, 64, 5)
assert big["forward_flops"] == 5 * base["forward_flops"], f'batch 5 should cost 5x forward, got {big["forward_flops"] / base["forward_flops"]}x'
assert big["params"] == base["params"], 'params must not depend on batch_size'
for out in (base, big):
    assert out["train_flops"] == 3 * out["forward_flops"], f'train_flops should be 3x forward_flops, got {out}'
""",
        },
        {
            "name": "Attention term grows quadratically with seq_len",
            "code": """
import torch
L, d, dff, V, B = 3, 64, 256, 1000, 2
# Weight matmuls are linear in tokens; the attention part is 4*L*B*s^2*d.
X = L * (4 * d * d + 2 * d * dff) + V * d
def attn_part(s):
    out = {fn}(L, d, dff, V, s, B)
    return out["forward_flops"] - 2 * (B * s) * X
a1 = attn_part(64)
a2 = attn_part(128)
assert a1 == 6291456, f'attention part at seq_len=64 should be 6291456, got {a1}'
assert a2 == 4 * a1, f'attention part should grow 4x when seq_len doubles, got {a2 / a1}x'
""",
        },
        {
            "name": "Random configs match reference formula",
            "code": """
import torch
torch.manual_seed(7)
for _ in range(6):
    L = int(torch.randint(1, 13, (1,)))
    d = 32 * int(torch.randint(1, 17, (1,)))
    dff = d * int(torch.randint(2, 5, (1,)))
    V = int(torch.randint(500, 80000, (1,)))
    s = int(torch.randint(1, 513, (1,)))
    B = int(torch.randint(1, 9, (1,)))
    tied = bool(torch.randint(0, 2, (1,)))
    out = {fn}(L, d, dff, V, s, B, tied_embeddings=tied)
    T = B * s
    p_ref = V * d + (0 if tied else V * d) + L * (4 * d * d + 2 * d * dff + 4 * d) + 2 * d
    f_ref = 2 * T * (L * (4 * d * d + 2 * d * dff) + V * d) + 4 * L * B * s * s * d
    cfg = f'L={L} d={d} d_ff={dff} V={V} s={s} B={B} tied={tied}'
    assert out["params"] == p_ref, f'{cfg}: params {out["params"]} != {p_ref}'
    assert out["forward_flops"] == f_ref, f'{cfg}: forward_flops {out["forward_flops"]} != {f_ref}'
    assert out["train_flops"] == 3 * f_ref, f'{cfg}: train_flops {out["train_flops"]} != {3 * f_ref}'
""",
        },
    ],
    "solution": '''def transformer_flops(n_layers, d_model, d_ff, vocab_size, seq_len, batch_size, tied_embeddings=True):
    d, V, L = d_model, vocab_size, n_layers
    tokens = batch_size * seq_len

    # Parameters
    embed = V * d
    lm_head = 0 if tied_embeddings else V * d    # tied head reuses the embedding matrix
    per_layer = 4 * d * d + 2 * d * d_ff + 4 * d  # Q,K,V,O; MLP up/down; 2 LayerNorms (gamma, beta)
    params = embed + lm_head + L * per_layer + 2 * d  # + final LayerNorm

    # Forward FLOPs (one multiply-add = 2 FLOPs)
    # Weight matmuls: every token goes through every layer and through the LM head
    weight_flops = 2 * tokens * (L * (4 * d * d + 2 * d * d_ff) + V * d)
    # Attention: QK^T and attn @ V over the full seq_len x seq_len score matrix
    attn_flops = L * 4 * batch_size * seq_len * seq_len * d
    forward = weight_flops + attn_flops

    # Backward pass costs about twice the forward pass
    return {"params": params, "forward_flops": forward, "train_flops": 3 * forward}''',
    "demo": """L, d, d_ff, V, s = 12, 768, 3072, 50257, 1024
out = transformer_flops(L, d, d_ff, V, s, batch_size=1)
print(f"GPT-2 small params (tied): {out['params'] / 1e6:.1f}M")
print(f"Forward FLOPs for one 1024-token sequence: {out['forward_flops']:.3e}")
print(f"Train FLOPs: {out['train_flops']:.3e}")

rule_of_thumb = 6 * out['params'] * s
print(f"6 * params * tokens = {rule_of_thumb:.3e}  (ratio to exact: {rule_of_thumb / out['train_flops']:.3f})")

untied = transformer_flops(L, d, d_ff, V, s, batch_size=1, tied_embeddings=False)
print(f"Untied params: {untied['params'] / 1e6:.1f}M, forward FLOPs unchanged: {untied['forward_flops'] == out['forward_flops']}")""",
}
