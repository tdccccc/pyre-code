# Mistake: clips the new value around the return instead of around the rollout-time value
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
    v_clipped = returns + torch.clamp(values - returns, -value_clip, value_clip)  # BUG: should be centered on old_values
    vf = 0.5 * torch.maximum((values - returns) ** 2, (v_clipped - returns) ** 2)
    vf_loss = masked_mean(vf)
    with torch.no_grad():
        clipfrac = masked_mean((pg_clipped > pg_unclipped).to(logps.dtype))
        approx_kl = masked_mean(old_logps - logps)
    return {
        "loss": pg_loss + vf_coef * vf_loss,
        "pg_loss": pg_loss,
        "vf_loss": vf_loss,
        "clipfrac": clipfrac,
        "approx_kl": approx_kl,
    }
