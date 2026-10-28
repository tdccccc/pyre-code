# Mistake: materializes the full (B, H, S) score tensor first and only then splits it, breaking the memory constraint
import torch


def flash_decoding(q, K, V, num_splits):
    B, H, D = q.shape
    scale = D ** -0.5

    scores = torch.einsum('bhd,bhsd->bhs', q, K) * scale  # BUG: full (B, H, S) scores
    s_chunks = torch.tensor_split(scores, num_splits, dim=-1)
    V_chunks = torch.tensor_split(V, num_splits, dim=2)

    outs, lses = [], []
    for s_c, Vc in zip(s_chunks, V_chunks):
        lse_i = torch.logsumexp(s_c, dim=-1)
        p = torch.exp(s_c - lse_i.unsqueeze(-1))
        outs.append(torch.einsum('bhs,bhsd->bhd', p, Vc))
        lses.append(lse_i)

    lses = torch.stack(lses, dim=0)
    lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - lse)
    out = (weights.unsqueeze(-1) * torch.stack(outs, dim=0)).sum(dim=0)
    return out, lse
