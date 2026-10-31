# Mistake: does not detach old_logps, so gradient leaks into the behaviour policy's log-probs
def gspo_loss(logps, old_logps, advantages, mask, clip_low=3e-4, clip_high=4e-4):
    m = mask.to(logps.dtype)
    adv = advantages.detach().to(logps.dtype)
    log_ratio = (logps - old_logps) * m
    seq_ratio = torch.exp(log_ratio.sum(dim=1) / m.sum(dim=1))
    clipped = torch.clamp(seq_ratio, 1 - clip_low, 1 + clip_high)
    surr = torch.minimum(seq_ratio * adv, clipped * adv)
    return -surr.mean()
