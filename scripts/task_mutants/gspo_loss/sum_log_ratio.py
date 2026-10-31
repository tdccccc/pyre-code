# Mistake: sums the token log-ratios without dividing by the length, so s_i is the raw sequence likelihood ratio
def gspo_loss(logps, old_logps, advantages, mask, clip_low=3e-4, clip_high=4e-4):
    m = mask.to(logps.dtype)
    adv = advantages.detach().to(logps.dtype)
    log_ratio = (logps - old_logps.detach()) * m
    seq_ratio = torch.exp(log_ratio.sum(dim=1))  # BUG: no length normalization
    clipped = torch.clamp(seq_ratio, 1 - clip_low, 1 + clip_high)
    surr = torch.minimum(seq_ratio * adv, clipped * adv)
    return -surr.mean()
