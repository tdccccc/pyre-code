# Mistake: uses r = log(pi / pi_ref) inside k3, which estimates the reverse direction and is no longer unbiased under samples from pi
def kl_estimate(logps, ref_logps, kind="k3"):
    r = ref_logps.detach() - logps
    if kind == "k1":
        return -r
    if kind == "k2":
        return 0.5 * r ** 2
    if kind == "k3":
        return torch.exp(-r) - 1 + r  # BUG: ratio inverted
    raise ValueError(f"unknown kind: {kind}")
