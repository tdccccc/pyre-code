# Mistake: failed extractions are not filtered out, so None can win the vote (and its score counts)
def select_answer(answers, scores=None, method="majority"):
    if method not in ("majority", "best_of_n", "weighted"):
        raise ValueError(f"unknown method: {method}")
    if method != "majority" and scores is None:
        raise ValueError(f"{method} needs scores")
    if scores is not None and len(scores) != len(answers):
        raise ValueError("scores and answers must have the same length")
    # BUG: every sample takes part, including those with answer None
    keep = list(range(len(answers)))
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
