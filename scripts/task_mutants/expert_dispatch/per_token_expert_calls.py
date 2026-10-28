# Mistake: calls each expert once per token instead of once on the batch of its accepted tokens
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
    for r in range(top_k):
        for n in range(N):
            e = int(top_idx[n, r])
            if fill[e] < capacity:
                fill[e] += 1
                accepted[e].append((n, gates[n, r]))
            else:
                num_dropped += 1

    out = torch.zeros_like(x)
    for e, slots in enumerate(accepted):
        for n, g in slots:  # BUG: one expert call per token
            y = experts[e](x[n:n + 1])
            out = out.index_add(0, torch.tensor([n], device=x.device), g * y)
    return out, num_dropped
