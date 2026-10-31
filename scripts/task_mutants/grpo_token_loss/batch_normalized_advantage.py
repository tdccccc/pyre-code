# Mistake: standardizes rewards over the whole batch instead of within each prompt group
def grpo_token_loss(logps, old_logps, ref_logps, rewards, mask, group_size,
                    clip_ratio=0.2, kl_coef=0.04, eps=1e-6):
    m = mask.to(logps.dtype)
    old_logps, ref_logps = old_logps.detach(), ref_logps.detach()
    r = rewards.detach().to(logps.dtype).view(-1, group_size)
    flat = r.view(-1)
    adv = ((flat - flat.mean()) / (flat.std() + eps)).view(-1, 1)  # BUG: should be per group
    ratio = torch.exp(logps - old_logps)
    surr = torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip_ratio, 1 + clip_ratio) * adv)
    log_r = ref_logps - logps
    kl = torch.exp(log_r) - log_r - 1
    token_loss = -(surr - kl_coef * kl)
    seq_loss = (token_loss * m).sum(dim=1) / m.sum(dim=1)
    return seq_loss.mean()
