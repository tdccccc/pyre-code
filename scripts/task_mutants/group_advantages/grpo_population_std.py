# Mistake: GRPO standardizes with the population std (unbiased=False) instead of the unbiased std
def group_advantages(rewards, group_size, method, eps=1e-6):
    if group_size < 2:
        raise ValueError("group_size must be at least 2")
    if rewards.numel() % group_size != 0:
        raise ValueError("the batch size must be a multiple of group_size")
    G = group_size
    r = rewards.detach().float().view(-1, G)
    mean = r.mean(dim=1, keepdim=True)

    if method == "grpo":
        adv = (r - mean) / (r.std(dim=1, keepdim=True, unbiased=False) + eps)
    elif method == "dr_grpo":
        adv = r - mean
    elif method == "rloo":
        adv = r - (r.sum(dim=1, keepdim=True) - r) / (G - 1)
    elif method == "reinforce_pp":
        x = (r - mean).view(-1)
        return (x - x.mean()) / (x.std() + eps)
    else:
        raise ValueError(f"unknown method: {method}")
    return adv.view(-1)
