# Mistake: drops the 1/2 in k2, doubling the estimate
def kl_estimate(logps, ref_logps, kind="k3"):
    r = ref_logps.detach() - logps
    if kind == "k1":
        return -r
    if kind == "k2":
        return r ** 2  # BUG: should be 0.5 * r ** 2
    if kind == "k3":
        return torch.exp(r) - 1 - r
    raise ValueError(f"unknown kind: {kind}")
