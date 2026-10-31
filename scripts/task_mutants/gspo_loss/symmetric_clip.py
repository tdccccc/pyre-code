# Mistake: uses clip_low on both sides of the clip range, ignoring clip_high
def gspo_loss(logps, old_logps, advantages, mask, clip_low=3e-4, clip_high=4e-4):
    m = mask.to(logps.dtype)
    adv = advantages.detach().to(logps.dtype)
    log_ratio = (logps - old_logps.detach()) * m
    seq_ratio = torch.exp(log_ratio.sum(dim=1) / m.sum(dim=1))
    clipped = torch.clamp(seq_ratio, 1 - clip_low, 1 + clip_low)
    surr = torch.minimum(seq_ratio * adv, clipped * adv)
    return -surr.mean()
