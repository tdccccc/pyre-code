# Mistake: counts every token whose ratio leaves [1-eps, 1+eps] as clipped, ignoring that with the other advantage sign the unclipped branch is used
def ppo_token_loss(logps, old_logps, advantages, values, old_values, returns, mask,
                   clip_ratio=0.2, value_clip=0.2, vf_coef=0.5):
    m = mask.to(logps.dtype)
    n_tokens = m.sum()

    def masked_mean(x):
        return (x * m).sum() / n_tokens

    old_logps, advantages = old_logps.detach(), advantages.detach()
    old_values, returns = old_values.detach(), returns.detach()
    ratio = torch.exp(logps - old_logps)
    pg_unclipped = -advantages * ratio
    pg_clipped = -advantages * torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
    pg_loss = masked_mean(torch.maximum(pg_unclipped, pg_clipped))
    v_clipped = old_values + torch.clamp(values - old_values, -value_clip, value_clip)
    vf = 0.5 * torch.maximum((values - returns) ** 2, (v_clipped - returns) ** 2)
    vf_loss = masked_mean(vf)
    with torch.no_grad():
        clipfrac = masked_mean(((ratio < 1 - clip_ratio) | (ratio > 1 + clip_ratio)).to(logps.dtype))  # BUG
        approx_kl = masked_mean(old_logps - logps)
    return {
        "loss": pg_loss + vf_coef * vf_loss,
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "clipfrac": clipfrac,
        "approx_kl": approx_kl,
    }
