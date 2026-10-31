# Mistake: k1 returns log(pi_ref / pi) instead of log(pi / pi_ref), so its expectation is -KL
def kl_estimate(logps, ref_logps, kind="k3"):
    r = ref_logps.detach() - logps
    if kind == "k1":
        return r  # BUG: sign flipped
    if kind == "k2":
        return 0.5 * r ** 2
    if kind == "k3":
        return torch.exp(r) - 1 - r
    raise ValueError(f"unknown kind: {kind}")
