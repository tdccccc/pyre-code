# Mistake: weighted voting averages the scores per answer instead of summing them, so one lucky high score beats a consistent majority
def select_answer(answers, scores=None, method="majority"):
    if method not in ("majority", "best_of_n", "weighted"):
        raise ValueError(f"unknown method: {method}")
    if method != "majority" and scores is None:
        raise ValueError(f"{method} needs scores")
    if scores is not None and len(scores) != len(answers):
        raise ValueError("scores and answers must have the same length")
    keep = [i for i, a in enumerate(answers) if a is not None]
    if not keep:
        return None
    if method == "best_of_n":
        best = keep[0]
        for i in keep[1:]:
            if scores[i] > scores[best]:
                best = i
        return answers[best]
    first_seen, total, count = {}, {}, {}
    for i in keep:
        a = answers[i]
        first_seen.setdefault(a, i)
        total[a] = total.get(a, 0) + (1 if method == "majority" else scores[i])
        count[a] = count.get(a, 0) + 1
    if method == "weighted":
        # BUG: mean score per answer, not the sum
        total = {a: total[a] / count[a] for a in total}
    return max(total, key=lambda a: (total[a], -first_seen[a]))
