# Mistake: causal mask built per local shard (block diagonal), so positions cannot see earlier shards
import torch


def ulysses_attention(q_shards, k_shards, v_shards, causal=False):
    P = len(q_shards)
    B, S_local, H, D = q_shards[0].shape
    H_local = H // P
    S = S_local * P
    scale = D ** -0.5
    device = q_shards[0].device

    def all_to_all_seq_to_head(shards):
        return [
            torch.cat([shards[s][:, :, r * H_local:(r + 1) * H_local] for s in range(P)], dim=1)
            for r in range(P)
        ]

    q_full = all_to_all_seq_to_head(q_shards)
    k_full = all_to_all_seq_to_head(k_shards)
    v_full = all_to_all_seq_to_head(v_shards)

    if causal:
        # BUG: uses the local shard length for every block instead of global positions
        local = torch.tril(torch.ones(S_local, S_local, dtype=torch.bool, device=device))
        mask = torch.block_diag(*([local] * P))
    else:
        mask = torch.ones(S, S, dtype=torch.bool, device=device)

    outs = []
    for q, k, v in zip(q_full, k_full, v_full):
        scores = torch.einsum('bihd,bjhd->bhij', q, k) * scale
        scores = scores.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum('bhij,bjhd->bihd', attn, v))

    result = []
    for s in range(P):
        chunks = [outs[r][:, s * S_local:(s + 1) * S_local] for r in range(P)]
        result.append(torch.cat(chunks, dim=2))
    return result
