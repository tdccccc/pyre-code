# Mistake: uses the desirable formula for undesirable completions too, so bad answers are also pushed above z0
def kto_loss(policy_logps, ref_logps, desirable, kl_policy_logps, kl_ref_logps,
             beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
    # Reference point: a batch estimate of KL(policy || ref), clamped at 0 and not trained through
    z0 = (kl_policy_logps - kl_ref_logps).mean().clamp(min=0).detach()
    r = policy_logps - ref_logps.detach()          # implicit reward
    # Desirable: push r above z0; undesirable: push r below z0
    good = desirable_weight * (1 - torch.sigmoid(beta * (r - z0)))
    bad = undesirable_weight * (1 - torch.sigmoid(beta * (r - z0)))  # BUG: sign should be (z0 - r)
    return torch.where(desirable.bool(), good, bad).mean()
