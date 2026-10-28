# Mistake: priority is token-major (token 0's 2nd choice beats token 1's 1st choice) instead of choice-rank-major
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
    num_dropped = 0
    # BUG: token loop outside the choice-rank loop
    for n in range(N):
        for r in range(top_k):
            e = int(top_idx[n, r])
            if fill[e] < capacity:
                fill[e] += 1
                accepted[e].append((n, gates[n, r]))
            else:
                num_dropped += 1

    out = torch.zeros_like(x)
    for e, slots in enumerate(accepted):
        if not slots:
            continue
        tokens = torch.tensor([n for n, _ in slots], device=x.device)
        w = torch.stack([g for _, g in slots])
        y = experts[e](x[tokens])
        out = out.index_add(0, tokens, w.unsqueeze(-1) * y)
    return out, num_dropped
