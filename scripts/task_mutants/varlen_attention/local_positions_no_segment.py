# Mistake: causal mask on per-sequence positions but without the same-sequence check, so tokens leak across sequences
import torch


def varlen_attention(q, k, v, cu_seqlens, causal=True):
    T, H, D = q.shape
    scale = D ** -0.5

    pos = torch.arange(T, device=q.device)
    seg = torch.searchsorted(cu_seqlens, pos, right=True) - 1
    local = pos - cu_seqlens[seg]  # position of each token within its own sequence
    if causal:
        mask = local[None, :] <= local[:, None]  # BUG: ignores seg, so earlier sequences leak in
    else:
        mask = torch.ones(T, T, dtype=torch.bool, device=q.device)
    scores = torch.einsum('ihd,jhd->hij', q, k) * scale
    scores = scores.masked_fill(~mask, float('-inf'))
    attn = torch.softmax(scores, dim=-1)
    return torch.einsum('hij,jhd->ihd', attn, v)
