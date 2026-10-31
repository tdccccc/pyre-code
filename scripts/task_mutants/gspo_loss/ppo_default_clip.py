# Mistake: keeps PPO's habitual 0.2 clip range as the default; GSPO's sequence ratios sit within ~1e-3 of 1, so nothing is ever clipped
def gspo_loss(logps, old_logps, advantages, mask, clip_low=0.2, clip_high=0.2):
    m = mask.to(logps.dtype)
    adv = advantages.detach().to(logps.dtype)
    log_ratio = (logps - old_logps.detach()) * m
    seq_ratio = torch.exp(log_ratio.sum(dim=1) / m.sum(dim=1))
    clipped = torch.clamp(seq_ratio, 1 - clip_low, 1 + clip_high)
    surr = torch.minimum(seq_ratio * adv, clipped * adv)
    return -surr.mean()
