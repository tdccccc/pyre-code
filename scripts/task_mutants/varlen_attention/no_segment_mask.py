# Mistake: causal mask over the whole packed batch, ignoring sequence boundaries so tokens attend across sequences
import torch


def varlen_attention(q, k, v, cu_seqlens, causal=True):
    T, H, D = q.shape
    scale = D ** -0.5

    mask = torch.ones(T, T, dtype=torch.bool, device=q.device)  # BUG: no same-sequence check
    if causal:
        mask = torch.tril(mask)
    scores = torch.einsum('ihd,jhd->hij', q, k) * scale
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum('hij,jhd->ihd', attn, v)
