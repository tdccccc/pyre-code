# Mistake: min-p compares probabilities to min_p directly instead of min_p * max(probs)
from collections import Counter


def apply_sampling_penalties(logits, prev_tokens, repetition_penalty=1.0, frequency_penalty=0.0, presence_penalty=0.0, temperature=1.0, min_p=0.0):
    logits = logits.clone()
    counts = Counter(prev_tokens)

    if counts:
        ids = torch.tensor(list(counts.keys()), dtype=torch.long)
        cnt = torch.tensor(list(counts.values()), dtype=logits.dtype)
        vals = logits[ids]
        vals = torch.where(vals > 0, vals / repetition_penalty, vals * repetition_penalty)
        vals = vals - (frequency_penalty * cnt + presence_penalty)
        logits[ids] = vals

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    probs = torch.where(probs < min_p, torch.zeros_like(probs), probs)
    return probs / probs.sum()
