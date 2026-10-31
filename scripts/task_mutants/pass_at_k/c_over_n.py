# Mistake: reports the fraction of correct samples for every k, which is pass@1, not pass@k
def pass_at_k(num_samples, num_correct, k):
    if len(num_samples) != len(num_correct):
        raise ValueError("num_samples and num_correct must have the same length")
    for n in num_samples:
        if k > n:
            raise ValueError(f"k={k} is larger than n={n}")
    # BUG: ignores k
    return sum(c / n for n, c in zip(num_samples, num_correct)) / len(num_samples)
