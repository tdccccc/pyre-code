"""Partition Equal Subset Sum (LeetCode 416)."""

import random


def _can_partition(nums: list[int]) -> bool:
    # Set of reachable subset sums, a different representation from the reference's bitset
    total = sum(nums)
    if total % 2:
        return False
    target = total // 2
    sums = {0}
    for x in nums:
        sums |= {s + x for s in sums if s + x <= target}
    return target in sums


def _large_case():
    # Every number is even and the total is 2 (mod 4), so half the total is odd and no subset
    # can reach it. The answer is False, and a solver that only checks the parity of the total
    # gets it wrong
    rng = random.Random(416)
    nums = [2 * rng.randint(1, 50) for _ in range(200)]
    if sum(nums) % 4 != 2:
        nums[-1] += 2 if nums[-1] <= 98 else -2
    return {"args": [nums], "expected": _can_partition(nums)}


def _random_case(rng):
    # Half the arrays have one number larger than the rest added together, plus an even gap so
    # the total is even. The answer is False, but a solver that lets a number repeat can still
    # reach half the total from the small numbers. The other arrays are random
    small = [rng.randint(1, 5) for _ in range(rng.randint(1, 4))]
    if rng.random() < 0.5:
        nums = small + [sum(small) + 2 * rng.randint(1, 4)]
    else:
        nums = [rng.randint(1, 10) for _ in range(rng.randint(1, 8))]
    rng.shuffle(nums)
    return {"args": [nums]}


TASK = {
    "title": "Partition Equal Subset Sum",
    "title_zh": "分割等和子集",
    "difficulty": "Medium",
    "leetcode_id": 416,
    "leetcode_slug": "partition-equal-subset-sum",
    "category": "Dynamic Programming",
    "function_name": "canPartition",
    "params": [("nums", "int[]")],
    "returns": "bool",
    "description_en": "Given a non-empty array `nums` of positive integers, decide whether it can be split into two groups with equal sums. Every element must go into exactly one of the two groups.\n\n**Example 1:** `nums = [3,1,1,2,2,1]` → `true` (3 + 1 + 1 = 2 + 2 + 1 = 5)\n\n**Example 2:** `nums = [1,2,5]` → `false` (the total is 8, but no group sums to 4)\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 200$\n- $1 \\le \\texttt{nums[i]} \\le 100$",
    "description_zh": "给定由正整数组成的非空数组 `nums`，判断能否把它分成两组，使两组的和相等。每个元素必须恰好属于其中一组。\n\n**示例 1：** `nums = [3,1,1,2,2,1]` → `true`（3 + 1 + 1 = 2 + 2 + 1 = 5）\n\n**示例 2：** `nums = [1,2,5]` → `false`（总和为 8，但没有任何一组的和是 4）\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 200$\n- $1 \\le \\texttt{nums[i]} \\le 100$",
    "hint": "Each group must sum to half the total, so the total must be even. Then ask whether some subset reaches that half: a 0/1 knapsack question.",
    "hint_zh": "每组的和都必须是总和的一半，所以总和必须是偶数。之后问题变成 0/1 背包：能否选出若干数字，使其和恰好是总和的一半。",
    "cases": [
        {"args": [[3, 1, 1, 2, 2, 1]], "expected": True},
        {"args": [[1, 2, 5]], "expected": False},
        {"args": [[1]], "expected": False},
        # Each number may be used once. A forward-filled DP reuses the first 1 and wrongly
        # reaches 3 (1 + 1 + 1 needs three 1s, but only two exist)
        {"args": [[1, 1, 4]], "expected": False},
        # Even total, yet no subset sums to 3
        {"args": [[2, 2, 2]], "expected": False},
        {"args": [[4, 4, 7, 1]], "expected": True},
        {"args": [[100, 100, 100, 100]], "expected": True},
        {"args": [[1, 2, 3, 4, 5, 6, 7]], "expected": True},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def canPartition(self, nums: List[int]) -> bool:
        total = sum(nums)
        if total % 2:
            return False
        # Bit s is set when some subset of the numbers seen so far sums to s
        reachable = 1
        for x in nums:
            reachable |= reachable << x
        return bool((reachable >> (total // 2)) & 1)''',
    "explanation": "Each group must sum to `total / 2`, so an odd total is an immediate `false`. Otherwise the question is whether some subset of the numbers sums to `target = total / 2`: a 0/1 knapsack. The usual table `dp[s]` (can a subset reach `s?`) is updated per number by `dp[s] |= dp[s - x]`, going from high `s` down to low, so each number is used at most once. Here the whole table is one Python integer: bit `s` is `dp[s]`, and adding `x` is `reachable |= reachable << x`, which does that update for every `s` at once.\n\nTime O(n · sum) bit operations, done in machine words by Python's big integers, space O(sum).",
    "explanation_zh": "每组的和都必须是 `total / 2`，所以总和为奇数时直接返回 `false`。否则问题变成：能否选出若干数字，和恰为 `target = total / 2`，这是 0/1 背包。常见的表 `dp[s]`（能否凑出和 `s`）在加入一个数 `x` 时按 `dp[s] |= dp[s - x]` 更新，并且 `s` 要从大到小遍历，保证每个数只用一次。这里整张表用一个 Python 整数表示：第 `s` 位即 `dp[s]`，加入 `x` 就是 `reachable |= reachable << x`，一次操作更新所有 `s`。\n\n时间 O(n · sum) 次位运算（Python 大整数按机器字并行完成），空间 O(sum)。",
}
