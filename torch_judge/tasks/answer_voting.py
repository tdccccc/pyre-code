"""Majority / Best-of-N / Weighted Voting task."""

TASK = {
    "title": "Majority / Best-of-N / Weighted Voting",
    "title_zh": "答案投票：多数 / Best-of-N / 加权",
    "difficulty": "Easy",
    "description_en": "Pick one final answer out of $N$ sampled solutions to the same question. This is the simplest form of test-time scaling: sample several reasoning paths, extract each final answer, then choose by **majority vote** (self-consistency), by the **best reward-model score** (Best-of-N), or by **reward-weighted voting**.\n\n**Signature:** `select_answer(answers, scores=None, method=\"majority\") -> str | None`\n\n**Parameters:**\n- `answers` — list of $N$ extracted final answers (already normalized strings); an entry is `None` when no answer could be extracted from that sample\n- `scores` — list of $N$ floats from a reward model or verifier (one per sample), or `None`. Scores are arbitrary floats and may be negative (reward models often output raw logits).\n- `method` — `\"majority\"`, `\"best_of_n\"` or `\"weighted\"`\n\n**Returns:** the selected answer string, or `None` when every entry of `answers` is `None`\n\n**Rules:**\n- Samples whose answer is `None` are ignored completely, together with their scores.\n- `\"majority\"`: the answer that occurs most often. `scores` is not used.\n- `\"best_of_n\"`: the answer of the sample with the highest score.\n- `\"weighted\"`: for each distinct answer, sum the scores of the samples that gave it; return the answer with the largest sum:\n$$\n\\hat a = \\arg\\max_{a} \\sum_{i:\\ \\text{answer}_i = a} \\text{score}_i\n$$\n- **Ties** (equal counts, equal scores or equal sums) go to the answer that appears **first** in `answers`.\n\n**Errors (`ValueError`):**\n- `method` is `\"best_of_n\"` or `\"weighted\"` and `scores` is `None`\n- `scores` is given and its length differs from `len(answers)` — for every method, including `\"majority\"`\n- unknown `method`\n\nArguments are validated first, so these errors are raised even when every answer is `None`.\n\n**Constraints:**\n- Do not modify `answers` or `scores`.",
    "description_zh": "从同一道题的 $N$ 个采样解答中选出一个最终答案。这是最简单的推理时扩展（test-time scaling）：采样多条推理路径，抽出各自的最终答案，然后用**多数投票**（self-consistency）、**奖励模型最高分**（Best-of-N）或**按奖励加权投票**来选。\n\n**签名:** `select_answer(answers, scores=None, method=\"majority\") -> str | None`\n\n**参数:**\n- `answers` — $N$ 个抽取出的最终答案（已规范化的字符串）；某个样本抽不出答案时该项为 `None`\n- `scores` — $N$ 个浮点分数，来自奖励模型或验证器（每个样本一个），或为 `None`。分数是任意浮点数，可以为负（奖励模型常直接输出 logit）。\n- `method` — `\"majority\"`、`\"best_of_n\"` 或 `\"weighted\"`\n\n**返回:** 选中的答案字符串；当 `answers` 全部为 `None` 时返回 `None`\n\n**规则:**\n- 答案为 `None` 的样本完全忽略，它的分数也一并忽略。\n- `\"majority\"`：出现次数最多的答案，不使用 `scores`。\n- `\"best_of_n\"`：分数最高的那个样本的答案。\n- `\"weighted\"`：对每个不同的答案，把给出它的样本的分数相加，返回总和最大的答案：\n$$\n\\hat a = \\arg\\max_{a} \\sum_{i:\\ \\text{answer}_i = a} \\text{score}_i\n$$\n- **平局**（次数、分数或总和相等）时，选在 `answers` 中**最先出现**的那个答案。\n\n**错误（`ValueError`）:**\n- `method` 为 `\"best_of_n\"` 或 `\"weighted\"` 而 `scores` 为 `None`\n- 给了 `scores` 但长度与 `len(answers)` 不同——对所有方法都成立，包括 `\"majority\"`\n- 未知的 `method`\n\n先校验参数再处理答案，所以即使所有答案都是 `None`，上述错误也照样抛出。\n\n**约束:**\n- 不得修改 `answers` 和 `scores`。",
    "function_name": "select_answer",
    "hint": "keep = [i for i, a in enumerate(answers) if a is not None]\nif not keep: return None\norder = {}  # answer -> index of its first appearance\nmajority: count per answer;  weighted: sum of scores per answer\npick the max of (value, -first_index) so ties go to the earliest answer\nbest_of_n: scan kept samples, replace only on a strictly higher score",
    "hint_zh": "keep = [i for i, a in enumerate(answers) if a is not None]\nif not keep: return None\norder = {}  # 答案 -> 第一次出现的位置\nmajority：每个答案计数；weighted：每个答案的分数求和\n取 (值, -首次出现位置) 最大者，平局时自然选最早出现的答案\nbest_of_n：遍历保留的样本，只有分数严格更高时才替换",
    "tests": [
        {
            "name": "Majority vote returns the most frequent answer",
            "code": """
answers = ['12', '15', '12', '9', '12', '15']
out = {fn}(answers, method='majority')
assert out == '12', f'Expected 12, got {out!r}'
assert isinstance(out, str), f'Expected a str, got {type(out).__name__}'
# scores are ignored by majority voting
out2 = {fn}(['3', '4', '4'], [0.9, 0.1, 0.1], 'majority')
assert out2 == '4', f'Majority must ignore scores: expected 4, got {out2!r}'
""",
        },
        {
            "name": "Default method is majority voting",
            "code": """
out = {fn}(['b', 'a', 'a', 'c'])
assert out == 'a', f'select_answer(answers) should default to majority voting: expected a, got {out!r}'
""",
        },
        {
            "name": "Ties go to the answer that appears first",
            "code": """
# '7' and '3' both appear twice; '7' comes first even though '3' < '7'
out = {fn}(['7', '3', '3', '7', '1'], method='majority')
assert out == '7', f'Majority tie: expected 7 (first seen), got {out!r}'
out = {fn}(['zeta', 'alpha', 'alpha', 'zeta'], method='majority')
assert out == 'zeta', f'Majority tie: expected zeta (first seen), got {out!r}'
# Best-of-N tie on the top score
out = {fn}(['8', '5', '6'], [0.5, 0.75, 0.75], 'best_of_n')
assert out == '5', f'Best-of-N tie: expected 5 (first sample with the top score), got {out!r}'
# Weighted tie: both sums are 1.0
out = {fn}(['9', '2', '9', '2'], [0.25, 0.5, 0.75, 0.5], 'weighted')
assert out == '9', f'Weighted tie: expected 9 (first seen), got {out!r}'
""",
        },
        {
            "name": "Majority, Best-of-N and weighted voting can disagree",
            "code": """
answers = ['x', 'x', 'x', 'y', 'y', 'z']
scores = [0.125, 0.125, 0.125, 0.625, 0.5, 0.875]
# counts: x=3; sums: x=0.375, y=1.125, z=0.875; best single score: z
assert {fn}(answers, scores, 'majority') == 'x', 'majority should pick x'
assert {fn}(answers, scores, 'best_of_n') == 'z', 'best_of_n should pick z'
assert {fn}(answers, scores, 'weighted') == 'y', 'weighted should pick y (largest summed score)'
# Weighted voting sums, it does not average: a=1.2 beats b=0.9 although b's mean is higher
out = {fn}(['a', 'a', 'a', 'b'], [0.4, 0.4, 0.4, 0.9], 'weighted')
assert out == 'a', f'Weighted voting must sum scores per answer: expected a, got {out!r}'
""",
        },
        {
            "name": "Negative scores (raw logits) work",
            "code": """
answers = ['3', '8', '3', '5']
scores = [-2.5, -0.75, -1.5, -3.0]
out = {fn}(answers, scores, 'best_of_n')
assert out == '8', f'best_of_n with all-negative scores: expected 8 (score -0.75), got {out!r}'
# sums: '3' = -4.0, '8' = -0.75, '5' = -3.0
out = {fn}(answers, scores, 'weighted')
assert out == '8', f'weighted with negative scores: expected 8 (largest sum -0.75), got {out!r}'
out = {fn}(['1', '2'], [-1.0, 0.5], 'best_of_n')
assert out == '2', f'mixed-sign scores: expected 2, got {out!r}'
""",
        },
        {
            "name": "Failed extractions (None) are ignored",
            "code": """
out = {fn}([None, None, '4', None], method='majority')
assert out == '4', f'None must not be voted for: expected 4, got {out!r}'
out = {fn}(['4', None, '6'], [0.2, 0.99, 0.3], 'best_of_n')
assert out == '6', f'The score of a None sample must be ignored: expected 6, got {out!r}'
out = {fn}(['4', None, None, '6'], [0.5, 0.9, 0.9, 0.25], 'weighted')
assert out == '4', f'Weighted voting must skip None samples: expected 4, got {out!r}'
for m in ('majority', 'best_of_n', 'weighted'):
    out = {fn}([None, None], [0.1, 0.2], m)
    assert out is None, f'{m}: all-None answers should return None, got {out!r}'
""",
        },
        {
            "name": "Invalid arguments raise ValueError",
            "code": """
def raises(call):
    try:
        call()
    except ValueError:
        return True
    return False
assert raises(lambda: {fn}(['1', '2'], None, 'best_of_n')), 'best_of_n without scores should raise ValueError'
assert raises(lambda: {fn}(['1', '2'], None, 'weighted')), 'weighted without scores should raise ValueError'
assert raises(lambda: {fn}(['1', '2'], [0.5], 'weighted')), 'scores of the wrong length should raise ValueError'
assert raises(lambda: {fn}(['1', '2'], [0.5, 0.5], 'median')), 'an unknown method should raise ValueError'
assert raises(lambda: {fn}(['1', '2'], [0.5, 0.5, 0.5], 'majority')), 'a scores length mismatch should raise ValueError for majority too'
# Validation comes before the all-None shortcut
assert raises(lambda: {fn}([None, None], None, 'best_of_n')), 'best_of_n without scores should raise even if every answer is None'
assert raises(lambda: {fn}([None, None], [0.5], 'majority')), 'a length mismatch should raise even if every answer is None'
""",
        },
        {
            "name": "Inputs are not modified",
            "code": """
answers = ['3', None, '3', '5', '5', '5']
scores = [0.9, 0.1, 0.8, 0.2, 0.3, 0.1]
a0, s0 = list(answers), list(scores)
for m in ('majority', 'best_of_n', 'weighted'):
    {fn}(answers, scores, m)
assert answers == a0, 'answers was modified'
assert scores == s0, 'scores was modified'
""",
        },
    ],
    "solution": '''def select_answer(answers, scores=None, method="majority"):
    if method not in ("majority", "best_of_n", "weighted"):
        raise ValueError(f"unknown method: {method}")
    if method != "majority" and scores is None:
        raise ValueError(f"{method} needs scores")
    if scores is not None and len(scores) != len(answers):
        raise ValueError("scores and answers must have the same length")

    # Failed extractions take no part in the vote, and neither do their scores
    keep = [i for i, a in enumerate(answers) if a is not None]
    if not keep:
        return None

    if method == "best_of_n":
        best = keep[0]
        for i in keep[1:]:
            if scores[i] > scores[best]:  # strict: ties keep the earlier sample
                best = i
        return answers[best]

    first_seen = {}
    total = {}
    for i in keep:
        a = answers[i]
        first_seen.setdefault(a, i)
        total[a] = total.get(a, 0) + (1 if method == "majority" else scores[i])
    # Largest total wins; on a tie the answer seen first wins
    return max(total, key=lambda a: (total[a], -first_seen[a]))''',
    "demo": """# 8 sampled solutions; one failed to produce an answer (None)
answers = ['12', '15', '12', None, '15', '7', '12', '7']
scores = [0.30, 0.91, 0.25, 0.99, 0.88, 0.97, 0.20, 0.10]
# majority -> 12 (3 votes); best_of_n -> 7 (0.97, the None's 0.99 is ignored);
# weighted -> 15 (0.91 + 0.88 = 1.79 beats 7: 1.07 and 12: 0.75)

for method in ('majority', 'best_of_n', 'weighted'):
    print(f'{method:>9}:', select_answer(answers, scores, method))""",

}
