# Mistake: clips symmetrically with eps_low on both sides, losing the wider upper bound of clip-higher
def dapo_loss(logps, old_logps, rewards, lengths, mask, group_size, max_len, cache_len,
              eps_low=0.2, eps_high=0.28, eps=1e-6):
    G = group_size
    old_logps = old_logps.detach()
    r = rewards.detach().to(logps.dtype)

    # 1. Dynamic sampling: a group whose raw rewards are all equal has zero advantage
    raw = r.view(-1, G)
    keep = raw.max(dim=1).values > raw.min(dim=1).values          # (P,)
    if not keep.any():
        return (logps * 0).sum()                                    # zero, still connected to logps

    # 2. Overlong shaping: 0 up to S, linear down to -1 at max_len, -1 beyond
    L = lengths.to(logps.dtype)
    soft = max_len - cache_len
    penalty = torch.where(L <= soft, torch.zeros_like(L),
                          torch.where(L <= max_len, (soft - L) / cache_len, -torch.ones_like(L)))
    shaped = (r + penalty).view(-1, G)

    # 3. Group-relative advantage on the shaped rewards (dropped groups are masked out below)
    adv = (shaped - shaped.mean(dim=1, keepdim=True)) / (shaped.std(dim=1, keepdim=True) + eps)
    adv = adv.view(-1, 1)

    # 4. Clip-higher: the upper bound 1 + eps_high is wider than the lower 1 - eps_low
    ratio = torch.exp(logps - old_logps)
    # BUG: the upper bound should be 1 + eps_high
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - eps_low, 1 + eps_low) * adv)

    # 5. Token-mean over every valid token of the kept rows
    m = mask.to(logps.dtype) * keep.repeat_interleave(G).to(logps.dtype)[:, None]
    return -(surr * m).sum() / m.sum()
