# Mistake: never validates scores, so best_of_n / weighted without scores (or with the wrong length) crash with TypeError/IndexError or silently use a prefix
def select_answer(answers, scores=None, method="majority"):
    if method not in ("majority", "best_of_n", "weighted"):
        raise ValueError(f"unknown method: {method}")
    keep = [i for i, a in enumerate(answers) if a is not None]
    if not keep:
        return None
    if method == "best_of_n":
        best = keep[0]
        for i in keep[1:]:
            if scores[i] > scores[best]:
                best = i
        return answers[best]
    first_seen, total = {}, {}
    for i in keep:
        a = answers[i]
        first_seen.setdefault(a, i)
        total[a] = total.get(a, 0) + (1 if method == "majority" else scores[i])
    return max(total, key=lambda a: (total[a], -first_seen[a]))
