# Mistake: applies the segment and causal mask after softmax and forgets to renormalize the rows
import torch


def varlen_attention(q, k, v, cu_seqlens, causal=True):
    T, H, D = q.shape
    scale = D ** -0.5

    pos = torch.arange(T, device=q.device)
    seg = torch.searchsorted(cu_seqlens, pos, right=True) - 1
    mask = seg[:, None] == seg[None, :]
    if causal:
        mask = mask & (pos[None, :] <= pos[:, None])
    scores = torch.einsum('ihd,jhd->hij', q, k) * scale
    attn = torch.softmax(scores, dim=-1)
    attn = attn * mask  # BUG: rows no longer sum to 1
    return torch.einsum('hij,jhd->ihd', attn, v)
