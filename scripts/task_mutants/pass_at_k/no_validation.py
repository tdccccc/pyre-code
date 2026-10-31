# Mistake: skips input validation, so k > n silently counts as solved and mismatched lists are truncated by zip
def pass_at_k(num_samples, num_correct, k):
    total = 0.0
    for n, c in zip(num_samples, num_correct):
        # BUG: no check that k <= n; n - c < k then returns 1.0
        if n - c < k:
            total += 1.0
            continue
        prob_all_wrong = 1.0
        for i in range(n - c + 1, n + 1):
            prob_all_wrong *= 1.0 - k / i
        total += 1.0 - prob_all_wrong
    return total / len(num_samples)
