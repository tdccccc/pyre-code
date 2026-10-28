# Mistake: renormalizes the surviving gates after dropping, so a token that lost a choice has its remaining gate boosted to 1
import math


def moe_dispatch(x, router_logits, experts, top_k=2, capacity_factor=1.25):
    N, D = x.shape
    E = router_logits.shape[-1]
    capacity = math.ceil(capacity_factor * N * top_k / E)

    probs = torch.softmax(router_logits, dim=-1)
    top_vals, top_idx = probs.topk(top_k, dim=-1)
    gates = top_vals / top_vals.sum(dim=-1, keepdim=True)

    fill = [0] * E
    accepted = [[] for _ in range(E)]
    keep = torch.zeros(N, top_k)
    num_dropped = 0
    for r in range(top_k):
        for n in range(N):
            e = int(top_idx[n, r])
            if fill[e] < capacity:
                fill[e] += 1
                keep[n, r] = 1.0
                accepted[e].append((n, r))
            else:
                num_dropped += 1

    # BUG: renormalize the surviving gates of each token
    kept = gates * keep
    gates = kept / kept.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    out = torch.zeros_like(x)
    for e, slots in enumerate(accepted):
        if not slots:
            continue
        tokens = torch.tensor([n for n, _ in slots], device=x.device)
        w = torch.stack([gates[n, r] for n, r in slots])
        y = experts[e](x[tokens])
        out = out.index_add(0, tokens, w.unsqueeze(-1) * y)
    return out, num_dropped
