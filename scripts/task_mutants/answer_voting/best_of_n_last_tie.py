# Mistake: Best-of-N replaces the current best on equal scores, so a tie goes to the last sample instead of the first
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
            # BUG: >= lets a later sample with the same score take over
            if scores[i] >= scores[best]:
                best = i
        return answers[best]
    first_seen, total = {}, {}
    for i in keep:
        a = answers[i]
        first_seen.setdefault(a, i)
        total[a] = total.get(a, 0) + (1 if method == "majority" else scores[i])
    return max(total, key=lambda a: (total[a], -first_seen[a]))
