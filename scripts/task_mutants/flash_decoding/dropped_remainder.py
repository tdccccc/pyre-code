# Mistake: uses chunk size S // num_splits, so the trailing S % num_splits tokens are silently dropped
import torch


def flash_decoding(q, K, V, num_splits):
    B, H, D = q.shape
    S = K.shape[2]
    scale = D ** -0.5
    size = S // num_splits  # BUG: remainder tokens never enter any chunk

    outs, lses = [], []
    for i in range(num_splits):
        Kc = K[:, :, i * size:(i + 1) * size]
        Vc = V[:, :, i * size:(i + 1) * size]
        scores = torch.einsum('bhd,bhsd->bhs', q, Kc) * scale
        lse_i = torch.logsumexp(scores, dim=-1)
        p = torch.exp(scores - lse_i.unsqueeze(-1))
        outs.append(torch.einsum('bhs,bhsd->bhd', p, Vc))
        lses.append(lse_i)

    lses = torch.stack(lses, dim=0)
    lse = torch.logsumexp(lses, dim=0)
    weights = torch.exp(lses - lse)
    out = (weights.unsqueeze(-1) * torch.stack(outs, dim=0)).sum(dim=0)
    return out, lse
