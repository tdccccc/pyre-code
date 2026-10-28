# Mistake: chunks are views of the caller's tensors and are accumulated in place, so the inputs are modified
def ring_allreduce(tensors):
    W = len(tensors)
    chunks = [list(t.tensor_split(W)) for t in tensors]
    log = []

    for t in range(W - 1):
        sent = [chunks[r][(r - t) % W] for r in range(W)]
        entry = []
        for r in range(W):
            dst, c = (r + 1) % W, (r - t) % W
            chunks[dst][c].add_(sent[r])
            entry.append((r, dst, c))
        log.append(entry)

    for t in range(W - 1):
        sent = [chunks[r][(r + 1 - t) % W] for r in range(W)]
        entry = []
        for r in range(W):
            dst, c = (r + 1) % W, (r + 1 - t) % W
            chunks[dst][c].copy_(sent[r])
            entry.append((r, dst, c))
        log.append(entry)

    outputs = [torch.cat(chunks[r]) for r in range(W)]
    return outputs, log
