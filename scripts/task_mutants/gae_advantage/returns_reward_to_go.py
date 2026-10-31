# Mistake: uses the discounted reward-to-go as the return instead of A_t + V_t, so the critic target ignores lambda
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
        returns = torch.zeros_like(values)
        running = torch.zeros(B, dtype=values.dtype)
        togo = torch.zeros(B, dtype=values.dtype)
        for t in reversed(range(T)):
            running = deltas[:, t] + gamma * lam * running
            togo = rewards[:, t] + gamma * togo  # BUG: Monte-Carlo return
            advantages[:, t] = running
            returns[:, t] = togo
        return advantages * m, returns * m
