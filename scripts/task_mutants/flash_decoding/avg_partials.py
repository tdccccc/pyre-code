# Mistake: averages the partial outputs uniformly instead of weighting each by exp(lse_i - lse)
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

    lse = torch.logsumexp(torch.stack(lses, dim=0), dim=0)
    out = torch.stack(outs, dim=0).mean(dim=0)  # BUG: should weight by exp(lse_i - lse)
    return out, lse
