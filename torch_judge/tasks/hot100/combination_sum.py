"""Combination Sum (LeetCode 39)."""

import random


def _combinations(candidates: list, target: int) -> list:
    # Depth-first search over sorted candidates; written only to produce the expected answers
    ordered = sorted(candidates)
    out, path = [], []

    def go(start: int, remaining: int) -> None:
        if remaining == 0:
            out.append(list(path))
            return
        for j in range(start, len(ordered)):
            if ordered[j] > remaining:
                break
            path.append(ordered[j])
            go(j, remaining - ordered[j])
            path.pop()

    go(0, target)
    return out


def _large_case():
    # At the constraint limits: 20 candidates and a target of 40, with small candidates so the
    # answer is large (about 5800 combinations). The output size is the cost, so no search
    # approach is at risk of timing out here.
    rng = random.Random(39)
    candidates = rng.sample(range(2, 22), 20)
    return {"args": [candidates, 40], "expected": _combinations(candidates, 40)}


def _random_case(rng):
    # Small candidates and targets keep several candidates and repeats in play
    return {"args": [rng.sample(range(2, 9), rng.randint(1, 4)), rng.randint(1, 14)]}


TASK = {
    "title": "Combination Sum",
    "title_zh": "组合总和",
    "difficulty": "Medium",
    "leetcode_id": 39,
    "leetcode_slug": "combination-sum",
    "category": "Backtracking",
    "function_name": "combinationSum",
    "params": [("candidates", "int[]"), ("target", "int")],
    "returns": "int[][]",
    "check": "unordered_nested",
    "description_en": """You are given `candidates`, a list of distinct integers, and an integer `target`. Find every combination of candidates whose values add up to exactly `target`. A candidate may be used any number of times. Two combinations count as the same when they use the same values the same number of times, so `[2,3]` and `[3,2]` are one combination. Return the combinations in any order.

**Example 1:** `candidates = [3,4,5]`, `target = 8` → `[[3,5],[4,4]]`

**Example 2:** `candidates = [6,9]`, `target = 4` → `[]`

**Constraints:**
- `1 <= candidates.length <= 20`
- `2 <= candidates[i] <= 40`
- All values in `candidates` are distinct
- `1 <= target <= 40`""",
    "description_zh": """给定一个由互不相同的整数组成的列表 `candidates` 和一个整数 `target`。找出所有和为 `target` 的组合。每个数字可以被使用任意多次。如果两个组合使用的数字相同、且每个数字的次数也相同，就视为同一个组合，例如 `[2,3]` 和 `[3,2]` 是同一个。返回的组合顺序任意。

**示例 1：** `candidates = [3,4,5]`，`target = 8` → `[[3,5],[4,4]]`

**示例 2：** `candidates = [6,9]`，`target = 4` → `[]`

**约束：**
- `1 <= candidates.length <= 20`
- `2 <= candidates[i] <= 40`
- `candidates` 中的值互不相同
- `1 <= target <= 40`""",
    "hint": "Choose the candidates in a fixed order (a start index), so each combination is generated once. Allow reusing the current candidate by passing the same index on, not the next one.",
    "hint_zh": "按固定顺序选数（用起始下标控制），让每个组合只生成一次。允许重复使用当前数字时，递归传入的是当前下标，而不是下一个下标。",
    "cases": [
        {"args": [[3, 4, 5], 8], "expected": [[3, 5], [4, 4]]},
        {"args": [[6, 9], 4], "expected": []},
        {"args": [[2, 3, 6, 7], 7], "expected": [[2, 2, 3], [7]]},
        {"args": [[2, 3, 5], 8], "expected": [[2, 2, 2, 2], [2, 3, 3], [3, 5]]},
        # Reports [2,3] once; a search that tries every candidate at each step also reports [3,2]
        {"args": [[2, 3, 5], 5], "expected": [[2, 3], [5]]},
        {"args": [[7], 21], "expected": [[7, 7, 7]]},
        {"args": [[2, 5, 7], 12], "expected": [[2, 2, 2, 2, 2, 2], [2, 5, 5], [5, 7]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates = sorted(candidates)
        result, path = [], []

        def backtrack(start: int, remaining: int) -> None:
            if remaining == 0:
                result.append(path[:])
                return
            for i in range(start, len(candidates)):
                if candidates[i] > remaining:
                    break  # sorted, so every later candidate is too large as well
                path.append(candidates[i])
                backtrack(i, remaining - candidates[i])  # i, not i + 1: a candidate may be reused
                path.pop()

        backtrack(0, target)
        return result''',
    "explanation": "Backtracking over the candidates in sorted order. At each step, pick a candidate at or after the current index (so combinations come out in one canonical order and are never repeated), subtract it from the remaining sum, and recurse with the same index so the candidate can be reused. Stop a branch when the sum is exactly reached (record it) or when the next candidate overshoots (the rest of the sorted list overshoots too).\n\nTime is proportional to the number of answers times the combination length (the search visits no dead ends beyond the overshoot check), space O(target / min) for the recursion.",
    "explanation_zh": "对排序后的候选数做回溯。每一步从当前下标或其之后选一个数（这样每个组合只按一种固定顺序产生一次），从剩余和中减去它，然后用同一个下标递归，以允许重复使用。若剩余和恰好为 0，记录该组合；若下一个候选数已超过剩余和，由于已排序，后面的数也都超过，直接剪枝。\n\n时间与答案数量及组合长度成正比（除超出剪枝外没有死分支），递归空间 O(target / min)。",
}
