# Mistake: converts the binomial coefficients to floats before dividing, which overflows for large n
def pass_at_k(num_samples, num_correct, k):
    if len(num_samples) != len(num_correct):
        raise ValueError("num_samples and num_correct must have the same length")
    total = 0.0
    for n, c in zip(num_samples, num_correct):
        if k > n:
            raise ValueError(f"k={k} is larger than n={n}")
        # BUG: float(math.comb(10000, 500)) raises OverflowError
        total += 1.0 - float(math.comb(n - c, k)) / float(math.comb(n, k))
    return total / len(num_samples)
