# Mistake: checks for uniform groups after adding the length penalty, so all-wrong groups with different lengths are kept
def dapo_loss(logps, old_logps, rewards, lengths, mask, group_size, max_len, cache_len,
              eps_low=0.2, eps_high=0.28, eps=1e-6):
    G = group_size
    old_logps = old_logps.detach()
    r = rewards.detach().to(logps.dtype)

    # 1. Dynamic sampling: a group whose raw rewards are all equal has zero advantage
    L = lengths.to(logps.dtype)
    soft = max_len - cache_len
    penalty = torch.where(L <= soft, torch.zeros_like(L),
                          torch.where(L <= max_len, (soft - L) / cache_len, -torch.ones_like(L)))
    shaped = (r + penalty).view(-1, G)
    # BUG: the filter should look at the raw rewards, before shaping
    keep = shaped.max(dim=1).values > shaped.min(dim=1).values
    if not keep.any():
        return (logps * 0).sum()

    # 3. Group-relative advantage on the shaped rewards (dropped groups are masked out below)
    adv = (shaped - shaped.mean(dim=1, keepdim=True)) / (shaped.std(dim=1, keepdim=True) + eps)
    adv = adv.view(-1, 1)

    # 4. Clip-higher: the upper bound 1 + eps_high is wider than the lower 1 - eps_low
    ratio = torch.exp(logps - old_logps)
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps_low, 1 + eps_high) * adv)

    # 5. Token-mean over every valid token of the kept rows
    m = mask.to(logps.dtype) * keep.repeat_interleave(G).to(logps.dtype)[:, None]
    return -(surr * m).sum() / m.sum()
