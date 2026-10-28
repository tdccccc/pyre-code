"""Subarray Sum Equals K (LeetCode 560)."""

import random
from collections import Counter


def _count(nums: list[int], k: int) -> int:
    # Count pairs of prefix sums (earlier, later) whose difference is k
    seen = Counter({0: 1})
    pre = total = 0
    for x in nums:
        pre += x
        total += seen[pre - k]
        seen[pre] += 1
    return total


def _large_case():
    # 2·10⁴ numbers in [-1000, 1000], the constraint limits. Checking every subarray is about
    # 2·10⁸ additions in Python; the prefix-sum count is one pass.
    rng = random.Random(560)
    nums = [rng.randint(-1000, 1000) for _ in range(20000)]
    k = 0
    return {"args": [nums, k], "expected": _count(nums, k)}


def _random_case(rng):
    return {"args": [[rng.randint(-3, 3) for _ in range(rng.randint(1, 8))], rng.randint(-3, 5)]}


TASK = {
    "title": "Subarray Sum Equals K",
    "title_zh": "和为 K 的子数组",
    "difficulty": "Medium",
    "leetcode_id": 560,
    "leetcode_slug": "subarray-sum-equals-k",
    "category": "Substring",
    "function_name": "subarraySum",
    "params": [("nums", "int[]"), ("k", "int")],
    "returns": "int",
    "description_en": "Given an integer array `nums` and an integer `k`, return the number of contiguous, non-empty subarrays whose values add up to exactly `k`. The numbers may be negative.\n\n**Example 1:** `nums = [2,-1,3,0,-2], k = 2` → `4` (the subarrays `[2]`, `[-1,3]`, `[-1,3,0]` and the whole array)\n\n**Example 2:** `nums = [1,-1,1,1], k = 1` → `5`\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 2 \\times 10^{4}$\n- $-1000 \\le \\texttt{nums[i]} \\le 1000$\n- $-10^{7} \\le \\texttt{k} \\le 10^{7}$",
    "description_zh": "给定整数数组 `nums` 和整数 `k`，返回和恰好等于 `k` 的连续非空子数组的个数。数组中的数可以是负数。\n\n**示例 1：** `nums = [2,-1,3,0,-2], k = 2` → `4`（子数组 `[2]`、`[-1,3]`、`[-1,3,0]` 以及整个数组）\n\n**示例 2：** `nums = [1,-1,1,1], k = 1` → `5`\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 2 \\times 10^{4}$\n- $-1000 \\le \\texttt{nums[i]} \\le 1000$\n- $-10^{7} \\le \\texttt{k} \\le 10^{7}$",
    "hint": "Let `P` be the running prefix sum. A subarray sums to `k` exactly when an earlier prefix equals `P - k`. Count prefix sums in a hash map. A sliding window does not work here because of the negative numbers.",
    "hint_zh": "设 P 为当前的前缀和。某个子数组的和为 k，当且仅当某个更早的前缀和等于 P - k。用哈希表统计前缀和出现的次数。由于存在负数，滑动窗口在这里不适用。",
    "cases": [
        {"args": [[2, -1, 3, 0, -2], 2], "expected": 4},
        {"args": [[1, -1, 1, 1], 1], "expected": 5},
        # The subarray starting at index 0 needs the empty prefix counted; a map without {0: 1} gives 0
        {"args": [[3], 3], "expected": 1},
        {"args": [[0, 0, 0], 0], "expected": 6},
        {"args": [[1000, -1000, 1000], 1000], "expected": 3},
        {"args": [[1, 2, 3], 3], "expected": 2},
        # Negative values: a window that shrinks on "sum > k" never reaches [-1, 1]
        {"args": [[-1, -1, 1], 0], "expected": 1},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        seen = {0: 1}  # prefix sum -> how many earlier prefixes have it (the empty prefix is 0)
        prefix = count = 0
        for x in nums:
            prefix += x
            count += seen.get(prefix - k, 0)  # earlier prefixes P with prefix - P == k
            seen[prefix] = seen.get(prefix, 0) + 1  # counted after the lookup, so no self-pairing
        return count''',
    "explanation": "Prefix sums. With `P_0 = 0` and `P_j` the sum of the first `j` numbers, the subarray `nums[i..j-1]` sums to `P_j - P_i`. So for each `j`, add how many earlier prefixes equal `P_j - k`. A hash map from prefix value to its count gives that in O(1). The empty prefix `{0: 1}` covers subarrays that start at index 0, and the current prefix is stored only after it has been counted. Negative numbers are fine, which a sliding window cannot handle.\n\nTime O(n), space O(n).",
    "explanation_zh": "前缀和。令 `P_0 = 0`，`P_j` 为前 `j` 个数之和，则子数组 `nums[i..j-1]` 的和为 `P_j - P_i`。因此对每个 `j`，加上之前等于 `P_j - k` 的前缀和个数，哈希表记录每个前缀和的出现次数，即可 O(1) 完成。空前缀 `{0: 1}` 覆盖了从下标 0 开始的子数组；当前前缀和先计数、后入表。负数不影响这种方法，但会让滑动窗口失效。\n\n时间 O(n)，空间 O(n)。",
}
