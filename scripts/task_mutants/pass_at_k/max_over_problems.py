# Mistake: returns the best problem's pass@k instead of the mean over problems
def pass_at_k(num_samples, num_correct, k):
    if len(num_samples) != len(num_correct):
        raise ValueError("num_samples and num_correct must have the same length")
    terms = []
    for n, c in zip(num_samples, num_correct):
        if k > n:
            raise ValueError(f"k={k} is larger than n={n}")
        if n - c < k:
            terms.append(1.0)
            continue
        prob_all_wrong = 1.0
        for i in range(n - c + 1, n + 1):
            prob_all_wrong *= 1.0 - k / i
        terms.append(1.0 - prob_all_wrong)
    # BUG: the metric averages over problems
    return max(terms)
