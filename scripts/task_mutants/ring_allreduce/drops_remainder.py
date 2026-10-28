# Mistake: splits with L // W and silently drops the trailing L % W elements when L is not divisible by W
def ring_allreduce(tensors):
    W = len(tensors)
    L = tensors[0].numel()
    size = L // W
    chunks = [[t[c * size:(c + 1) * size].clone() for c in range(W)] for t in tensors]
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
        sent = [chunks[r][(r + 1 - t) % W] for r in range(W)]
        entry = []
        for r in range(W):
            dst, c = (r + 1) % W, (r + 1 - t) % W
            chunks[dst][c] = sent[r]
            entry.append((r, dst, c))
        log.append(entry)

    outputs = [torch.cat(chunks[r]) for r in range(W)]
    return outputs, log
