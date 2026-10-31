# Mistake: uses 1 - (1 - c/n)^k, which treats the k draws as independent with replacement and is biased low
def pass_at_k(num_samples, num_correct, k):
    if len(num_samples) != len(num_correct):
        raise ValueError("num_samples and num_correct must have the same length")
    total = 0.0
    for n, c in zip(num_samples, num_correct):
        if k > n:
            raise ValueError(f"k={k} is larger than n={n}")
        # BUG: should be C(n-c, k) / C(n, k), drawing without replacement
        total += 1.0 - (1.0 - c / n) ** k
    return total / len(num_samples)
