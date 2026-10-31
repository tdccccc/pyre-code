# Mistake: the GAE recursion decays by gamma only, dropping lambda (this is the Monte-Carlo advantage, not GAE)
def gae_advantage(scores, kl, values, mask, kl_coef=0.05, gamma=1.0, lam=0.95):
    with torch.no_grad():
        m = mask.to(values.dtype)
        B, T = values.shape
        lengths = m.sum(dim=1).long()
        rewards = -kl_coef * kl * m
        rewards[torch.arange(B), lengths - 1] += scores
        next_values = torch.zeros_like(values)
        next_values[:, :-1] = values[:, 1:] * m[:, 1:]
        deltas = rewards + gamma * next_values - values * m
        advantages = torch.zeros_like(values)
        running = torch.zeros(B, dtype=values.dtype)
        for t in reversed(range(T)):
            running = deltas[:, t] + gamma * running  # BUG: missing lam
            advantages[:, t] = running
        advantages = advantages * m
        returns = (advantages + values) * m
    return advantages, returns
