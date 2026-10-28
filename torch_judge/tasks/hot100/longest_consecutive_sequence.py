"""Longest Consecutive Sequence (LeetCode 128)."""

import random


def _longest(nums: list[int]) -> int:
    # Sort the distinct values and measure the runs of +1 steps
    best = run = 0
    prev = None
    for x in sorted(set(nums)):
        run = run + 1 if prev is not None and x == prev + 1 else 1
        best = max(best, run)
        prev = x
    return best


def _large_case():
    # 10^5 numbers: four runs of 20000 consecutive values, scattered filler and repeats, all
    # shuffled. A list-based "is x + 1 present?" check rescans the array at every step of every
    # run, which is far too slow; the set-based scan visits each value a constant number of times.
    rng = random.Random(128)
    nums = []
    for base in rng.sample(range(-10**9, 10**9 - 10**6), 4):
        nums.extend(range(base, base + 20000))
    nums.extend(rng.randint(-10**9, 10**9) for _ in range(15000))
    copies = [rng.choice(nums) for _ in range(5000)]
    nums.extend(copies)
    rng.shuffle(nums)
    return {"args": [nums], "expected": _longest(nums)}


def _random_case(rng):
    # One run of 3 to 5 consecutive values with one value repeated, shuffled. A repeat inside
    # the run restarts the mistake's streak, so it undercounts. Measured: about 10 of 20 cases
    # expose it
    start = rng.randint(0, 3)
    run_vals = list(range(start, start + rng.randint(3, 5)))
    nums = run_vals + [rng.choice(run_vals)]
    rng.shuffle(nums)
    return {"args": [nums]}


TASK = {
    "title": "Longest Consecutive Sequence",
    "title_zh": "最长连续序列",
    "difficulty": "Medium",
    "leetcode_id": 128,
    "leetcode_slug": "longest-consecutive-sequence",
    "category": "Hashing",
    "function_name": "longestConsecutive",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "Given an unsorted integer array `nums`, return the length of the longest run of consecutive integers that all appear in `nums`. A run is a set of integers that step by one, like `4, 5, 6`. Its members need not sit next to each other in the array, and a repeated value counts once.\n\n**Example 1:** `nums = [9,1,4,7,3,-1,0,5,8,-1,6]` → `7` (the run `3` through `9`)\n\n**Example 2:** `nums = [10,5,12,3,55,30,4,11,2]` → `4` (the run `2, 3, 4, 5`)\n\n**Constraints:**\n- `0 <= nums.length <= 10^5`\n- `-10^9 <= nums[i] <= 10^9`\n\n**Follow-up:** can you solve it in O(n) time?",
    "description_zh": "给定一个无序的整数数组 `nums`，返回 `nums` 中所有值都出现过的、能排成连续整数的最长一段的长度。一段连续整数形如 `4, 5, 6`，相邻两数相差 1。它们在数组中的位置不必相邻，重复出现的值只算一次。\n\n**示例 1：** `nums = [9,1,4,7,3,-1,0,5,8,-1,6]` → `7`（段 `3` 到 `9`）\n\n**示例 2：** `nums = [10,5,12,3,55,30,4,11,2]` → `4`（段 `2, 3, 4, 5`）\n\n**约束：**\n- `0 <= nums.length <= 10^5`\n- `-10^9 <= nums[i] <= 10^9`\n\n**进阶：** 你能设计一个 O(n) 时间的算法吗？",
    "hint": "Put every value in a hash set. A value starts a run only when `x - 1` is missing, so each run is walked once, from its start.",
    "hint_zh": "把所有值放进哈希集合。只有当 `x - 1` 不在集合里时，x 才是一段连续序列的起点，每段只从起点走一次。",
    "cases": [
        {"args": [[9, 1, 4, 7, 3, -1, 0, 5, 8, -1, 6]], "expected": 7},
        {"args": [[10, 5, 12, 3, 55, 30, 4, 11, 2]], "expected": 4},
        {"args": [[]], "expected": 0},
        # A repeated value must not break a run; a sort that restarts the count on a repeat gives 2
        {"args": [[1, 2, 2, 3]], "expected": 3},
        {"args": [[42]], "expected": 1},
        {"args": [[-5, -4, -2, -3]], "expected": 4},
        {"args": [[7, 7, 7]], "expected": 1},
        # Values near the limits: the two large ones are one step apart, the -10^9 stands alone
        {"args": [[1000000000, -1000000000, 999999999]], "expected": 2},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        seen = set(nums)  # O(1) membership; repeats collapse
        best = 0
        for x in seen:
            if x - 1 in seen:
                continue  # not the start of its run; the start walks the whole run
            length = 1
            while x + length in seen:
                length += 1
            best = max(best, length)
        return best''',
    "explanation": "A hash set gives O(1) membership and drops repeated values. Only a value whose predecessor `x - 1` is absent can start a run, so only those values walk upward. Every other value is skipped in O(1), and each value is passed at most twice overall: once by the outer loop and once while walking its run. That keeps the total work linear even though the walks themselves vary in length.\n\nTime O(n), space O(n).",
    "explanation_zh": "哈希集合提供 O(1) 的成员判断，并自动去掉重复值。只有前驱 `x - 1` 不在集合中的值（即一段的起点）才向上数，其余值直接跳过。每个值最多被外层循环访问一次、在所属段中被数到一次，所以总工作量与数组长度成正比。\n\n时间 O(n)，空间 O(n)。",
}
