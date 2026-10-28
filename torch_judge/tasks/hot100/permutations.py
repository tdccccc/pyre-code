"""Permutations (LeetCode 46)."""

import itertools
import random


def _large_case():
    # At the constraint limit: 8! = 40320 permutations. Backtracking handles that quickly, so
    # the real cost is the size of the output, not the search.
    rng = random.Random(46)
    nums = rng.sample(range(-10, 11), 8)
    return {"args": [nums], "expected": [list(p) for p in itertools.permutations(nums)]}


def _random_case(rng):
    return {"args": [rng.sample(range(-9, 10), rng.randint(1, 5))]}


TASK = {
    "title": "Permutations",
    "title_zh": "全排列",
    "difficulty": "Medium",
    "leetcode_id": 46,
    "leetcode_slug": "permutations",
    "category": "Backtracking",
    "function_name": "permute",
    "params": [("nums", "int[]")],
    "returns": "int[][]",
    "check": "unordered",
    "description_en": """Given a list `nums` of distinct integers, return every permutation of it: every way to arrange all of its elements in a row. The permutations may be returned in any order.

**Example 1:** `nums = [0,1]` → `[[0,1],[1,0]]`

**Example 2:** `nums = [1]` → `[[1]]`

**Constraints:**
- $1 \\le \\texttt{nums.length} \\le 8$
- $-10 \\le \\texttt{nums[i]} \\le 10$
- All integers in `nums` are distinct""",
    "description_zh": """给定一个由互不相同的整数组成的列表 `nums`，返回它的全部排列，即把所有元素排成一行的每一种方式。排列的输出顺序任意。

**示例 1：** `nums = [0,1]` → `[[0,1],[1,0]]`

**示例 2：** `nums = [1]` → `[[1]]`

**约束：**
- $1 \\le \\texttt{nums.length} \\le 8$
- $-10 \\le \\texttt{nums[i]} \\le 10$
- `nums` 中的整数互不相同""",
    "hint": "Build the permutation one position at a time, trying each number that is not used yet. Undo the choice when you backtrack, and save a copy of the list when it is full.",
    "hint_zh": "逐个位置填数，每次尝试一个还没用过的数字。回溯时撤销这次选择；排列填满时保存它的一份副本。",
    "cases": [
        {"args": [[0, 1]], "expected": [[0, 1], [1, 0]]},
        {"args": [[1]], "expected": [[1]]},
        # Zero and negative values: a solution that marks "used" with a value (instead of a flag) breaks here
        {"args": [[0, -3, 4]], "expected": [[0, -3, 4], [0, 4, -3], [-3, 0, 4], [-3, 4, 0], [4, 0, -3], [4, -3, 0]]},
        {"args": [[1, 2, 3]], "expected": [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]},
        {"args": [[10, -10]], "expected": [[10, -10], [-10, 10]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        result, path = [], []
        used = [False] * len(nums)

        def backtrack() -> None:
            if len(path) == len(nums):
                result.append(path[:])  # a copy: path keeps changing
                return
            for i, x in enumerate(nums):
                if not used[i]:
                    used[i] = True
                    path.append(x)
                    backtrack()
                    path.pop()
                    used[i] = False

        backtrack()
        return result''',
    "explanation": "Backtracking. Fill the permutation one position at a time; at each position try every number that is not in use yet, mark it as used, recurse, then undo the choice. When all positions are filled, save a copy of the current list. The undo step is what lets the same list be reused for every branch.\n\nTime O(n · n!), space O(n) for the recursion and the path (the output itself is n!·n).",
    "explanation_zh": "回溯法。逐个位置填数：每个位置尝试所有还没用过的数字，标记为已用、递归下一个位置，然后撤销选择。所有位置填满后，保存当前列表的一份副本。撤销这一步让同一个列表能被所有分支复用。\n\n时间 O(n · n!)，空间 O(n)（递归与路径；输出本身有 n!·n 个数）。",
}
