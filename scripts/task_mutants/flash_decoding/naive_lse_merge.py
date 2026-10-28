# Mistake: merges the chunk lses with log(sum(exp(lse_i))) and no max subtraction, so large scores overflow
import torch


def flash_decoding(q, K, V, num_splits):
    B, H, D = q.shape
    scale = D ** -0.5

    outs, lses = [], []
    for Kc, Vc in zip(torch.tensor_split(K, num_splits, dim=2), torch.tensor_split(V, num_splits, dim=2)):
        scores = torch.einsum('bhd,bhsd->bhs', q, Kc) * scale
        lse_i = torch.logsumexp(scores, dim=-1)
        p = torch.exp(scores - lse_i.unsqueeze(-1))
        outs.append(torch.einsum('bhs,bhsd->bhd', p, Vc))
        lses.append(lse_i)

    lses = torch.stack(lses, dim=0)
    lse = torch.log(torch.exp(lses).sum(dim=0))  # BUG: exp of large lse overflows in float32
    weights = torch.exp(lses - lse)
    out = (weights.unsqueeze(-1) * torch.stack(outs, dim=0)).sum(dim=0)
    return out, lse
