# Mistake: partial output is exp(s - max) @ V without dividing by the chunk's own softmax denominator
import torch


def flash_decoding(q, K, V, num_splits):
    B, H, D = q.shape
    scale = D ** -0.5

    outs, lses = [], []
    for Kc, Vc in zip(torch.tensor_split(K, num_splits, dim=2), torch.tensor_split(V, num_splits, dim=2)):
        scores = torch.einsum('bhd,bhsd->bhs', q, Kc) * scale
        lse_i = torch.logsumexp(scores, dim=-1)
        m = scores.max(dim=-1, keepdim=True).values
        p = torch.exp(scores - m)  # BUG: not divided by exp(lse_i - m)
        outs.append(torch.einsum('bhs,bhsd->bhd', p, Vc))
        lses.append(lse_i)

    lses = torch.stack(lses, dim=0)
    lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - lse)
    out = (weights.unsqueeze(-1) * torch.stack(outs, dim=0)).sum(dim=0)
    return out, lse
