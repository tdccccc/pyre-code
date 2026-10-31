# Mistake: clips each token ratio like GRPO and averages per response, instead of one clipped ratio per response
def gspo_loss(logps, old_logps, advantages, mask, clip_low=3e-4, clip_high=4e-4):
    m = mask.to(logps.dtype)
    adv = advantages.detach().to(logps.dtype)
    log_ratio = (logps - old_logps.detach()) * m
    ratio = torch.exp(logps - old_logps.detach())
    a = adv[:, None]
    tok = torch.minimum(ratio * a, torch.clamp(ratio, 1 - clip_low, 1 + clip_high) * a)
    return -((tok * m).sum(dim=1) / m.sum(dim=1)).mean()
