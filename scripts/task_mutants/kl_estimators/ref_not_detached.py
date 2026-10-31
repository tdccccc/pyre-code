# Mistake: lets gradients flow into the reference model's log-probs
def kl_estimate(logps, ref_logps, kind="k3"):
    r = ref_logps - logps  # BUG: reference should be detached
    if kind == "k1":
        return -r
    if kind == "k2":
        return 0.5 * r ** 2
    if kind == "k3":
        return torch.exp(r) - 1 - r
    raise ValueError(f"unknown kind: {kind}")
