# Mistake: all-gather forwards chunk (r - t) instead of (r + 1 - t), so the wrong chunk is overwritten
def ring_allreduce(tensors):
    W = len(tensors)
    chunks = [[c.clone() for c in t.tensor_split(W)] for t in tensors]
    log = []

    for t in range(W - 1):
        sent = [chunks[r][(r - t) % W] for r in range(W)]
        entry = []
        for r in range(W):
            dst, c = (r + 1) % W, (r - t) % W
            chunks[dst][c] = chunks[dst][c] + sent[r]
            entry.append((r, dst, c))
        log.append(entry)

    for t in range(W - 1):
        sent = [chunks[r][(r - t) % W] for r in range(W)]
        entry = []
        for r in range(W):
            dst, c = (r + 1) % W, (r - t) % W
            chunks[dst][c] = sent[r]
            entry.append((r, dst, c))
        log.append(entry)

    outputs = [torch.cat(chunks[r]) for r in range(W)]
    return outputs, log
