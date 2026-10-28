"""Subsets (LeetCode 78)."""

import itertools
import random


def _large_case():
    # At the constraint limit: 2^10 = 1024 subsets. Cheap to compute; the output is the work.
    rng = random.Random(78)
    nums = rng.sample(range(-10, 11), 10)
    expected = [list(c) for k in range(len(nums) + 1) for c in itertools.combinations(nums, k)]
    return {"args": [nums], "expected": expected}


def _random_case(rng):
    return {"args": [rng.sample(range(-5, 6), rng.randint(1, 4))]}


TASK = {
    "title": "Subsets",
    "title_zh": "子集",
    "difficulty": "Medium",
    "leetcode_id": 78,
    "leetcode_slug": "subsets",
    "category": "Backtracking",
    "function_name": "subsets",
    "params": [("nums", "int[]")],
    "returns": "int[][]",
    "check": "unordered_nested",
    "description_en": """Given a list `nums` of distinct integers, return every subset of it: each way of choosing some of its elements, including choosing none and choosing all of them. The subsets may come in any order, and so may the elements inside each subset.

**Example 1:** `nums = [4,5]` → `[[],[4],[5],[4,5]]` (in any order)

**Example 2:** `nums = [-2]` → `[[],[-2]]`

**Constraints:**
- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`
- All values in `nums` are distinct""",
    "description_zh": """给定一个由互不相同的整数组成的列表 `nums`，返回它的所有子集：即从中选取若干元素（可以一个都不选，也可以全选）的每一种选法。子集的顺序任意，子集内部元素的顺序也任意。

**示例 1：** `nums = [4,5]` → `[[],[4],[5],[4,5]]`（顺序任意）

**示例 2：** `nums = [-2]` → `[[],[-2]]`

**约束：**
- `1 <= nums.length <= 10`
- `-10 <= nums[i] <= 10`
- `nums` 中的值互不相同""",
    "hint": "Process the numbers one by one. Each new number doubles the answer: every subset found so far is kept, and a copy with the new number added is made.",
    "hint_zh": "逐个处理数字。每加入一个新数字，答案翻倍：已有的每个子集保留，并复制一份加上新数字。",
    "cases": [
        {"args": [[4, 5]], "expected": [[], [4], [5], [4, 5]]},
        {"args": [[-2]], "expected": [[], [-2]]},
        {"args": [[1, 2, 3]], "expected": [[], [1], [2], [3], [1, 2], [1, 3], [2, 3], [1, 2, 3]]},
        # Unsorted input with a zero: the empty subset must still be there
        {"args": [[9, -9, 0]], "expected": [[], [9], [-9], [0], [9, -9], [9, 0], [-9, 0], [9, -9, 0]]},
        # Forgetting the empty subset (starting from an empty result list) fails every case here
        {"args": [[3, 6]], "expected": [[], [3], [6], [3, 6]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        result = [[]]
        for x in nums:
            # Every subset found so far gets a copy that also contains x
            result += [subset + [x] for subset in result]
        return result''',
    "explanation": "Build the answer one element at a time. Start with the empty subset. For each new number, every subset already built is kept, and a copy with the number appended is added. After n numbers there are 2^n subsets, each choice of \"include or skip\" for every element exactly once.\n\nTime O(n · 2^n), space O(n · 2^n) for the output.",
    "explanation_zh": "逐个元素构造答案。从空子集开始，每加入一个新数字，已有的每个子集都保留，并新增一份加上该数字的副本。处理完 n 个数字后得到 2^n 个子集，每个元素恰好有「选 / 不选」两种情况被覆盖一次。\n\n时间 O(n · 2^n)，空间 O(n · 2^n)（输出本身的大小）。",
}
