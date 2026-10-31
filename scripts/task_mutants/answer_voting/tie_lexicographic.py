# Mistake: breaks ties by sorting the answers alphabetically instead of keeping the first one seen
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
    total = {}
    for i in keep:
        a = answers[i]
        total[a] = total.get(a, 0) + (1 if method == "majority" else scores[i])
    # BUG: max over the sorted keys returns the alphabetically smallest of the tied answers
    return max(sorted(total), key=lambda a: total[a])
