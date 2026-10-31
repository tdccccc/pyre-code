# Mistake: adds the KL term to the objective with the wrong sign, so the loss rewards drifting away from the reference model
def grpo_token_loss(logps, old_logps, ref_logps, rewards, mask, group_size,
                    clip_ratio=0.2, kl_coef=0.04, eps=1e-6):
    m = mask.to(logps.dtype)
    old_logps, ref_logps = old_logps.detach(), ref_logps.detach()
    r = rewards.detach().to(logps.dtype).view(-1, group_size)
    adv = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + eps)
    adv = adv.view(-1, 1)
    ratio = torch.exp(logps - old_logps)
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv)
    log_r = ref_logps - logps
    kl = torch.exp(log_r) - log_r - 1
    token_loss = -(surr + kl_coef * kl)  # BUG: KL is a penalty
    seq_loss = (token_loss * m).sum(dim=1) / m.sum(dim=1)
    return seq_loss.mean()
