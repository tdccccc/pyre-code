"""Longest Increasing Subsequence (LeetCode 300)."""

import random


def _length(nums: list[int]) -> int:
    # O(n²) dynamic programming: best[i] is the longest strictly increasing subsequence that
    # ends at i. Slower than the reference's O(n log n), but a different method
    best = [1] * len(nums)
    for i in range(len(nums)):
        for j in range(i):
            if nums[j] < nums[i] and best[j] + 1 > best[i]:
                best[i] = best[j] + 1
    return max(best)


def _large_case():
    # 2500 values drawn from only 601 distinct numbers, so repeats are everywhere and a
    # non-strict comparison gives a visibly wrong length. The O(n^2) DP also passes (about
    # 0.1 s at n <= 2500), so this checks correctness at scale, not the O(n log n) follow-up
    rng = random.Random(300)
    nums = [rng.randint(-300, 300) for _ in range(2500)]
    return {"args": [nums], "expected": _length(nums)}


def _random_case(rng):
    # Few distinct values, so equal neighbours (which must not both be counted) are common
    return {"args": [[rng.randint(0, 4) for _ in range(rng.randint(1, 10))]]}


TASK = {
    "title": "Longest Increasing Subsequence",
    "title_zh": "最长递增子序列",
    "difficulty": "Medium",
    "leetcode_id": 300,
    "leetcode_slug": "longest-increasing-subsequence",
    "category": "Dynamic Programming",
    "function_name": "lengthOfLIS",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "Given an integer array `nums`, return the length of its longest **strictly** increasing subsequence. A subsequence keeps the original order of the elements but may skip any of them, and each chosen element must be larger than the one before it. Equal values cannot both be used.\n\n**Example 1:** `nums = [6,1,4,2,8,3,9]` → `4` (for example $1,\\ 2,\\ 3,\\ 9$)\n\n**Example 2:** `nums = [7,7,7,7]` → `1`\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 2500$\n- $-10^{4} \\le \\texttt{nums[i]} \\le 10^{4}$\n\n**Follow-up:** can you get the answer in O(n log n) time?",
    "description_zh": "给定整数数组 `nums`，返回其最长**严格**递增子序列的长度。子序列保持元素的原有顺序，但可以跳过任意元素；选中的每个元素都必须比前一个大。相等的值不能同时选入。\n\n**示例 1：** `nums = [6,1,4,2,8,3,9]` → `4`（例如 $1,\\ 2,\\ 3,\\ 9$）\n\n**示例 2：** `nums = [7,7,7,7]` → `1`\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 2500$\n- $-10^{4} \\le \\texttt{nums[i]} \\le 10^{4}$\n\n**进阶：** 你能否把时间复杂度降到 O(n log n)？",
    "hint": "Keep `tails[k]`, the smallest value that can end an increasing subsequence of length k + 1. The list stays sorted, so binary-search where each new number belongs in it.",
    "hint_zh": "维护 `tails[k]`：能结束一个长度为 k + 1 的递增子序列的最小值。这个列表始终有序，所以每个新数字都可以在其中二分查找自己的位置。",
    "cases": [
        {"args": [[6, 1, 4, 2, 8, 3, 9]], "expected": 4},
        {"args": [[7, 7, 7, 7]], "expected": 1},
        {"args": [[5]], "expected": 1},
        {"args": [[1, 2, 3, 4, 5]], "expected": 5},
        {"args": [[5, 4, 3, 2, 1]], "expected": 1},
        # Equal values cannot both be used; counting them would give 5
        {"args": [[2, 2, 3, 3, 4]], "expected": 3},
        {"args": [[3, -1, 0, -5, 2, -4, 9]], "expected": 4},
        {"args": [[0, 2, 1, 3, 2, 4]], "expected": 4},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from bisect import bisect_left


class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        # tails[k]: the smallest value that can end an increasing subsequence of length k + 1
        tails: List[int] = []
        for x in nums:
            k = bisect_left(tails, x)  # first tail >= x; x belongs at that spot
            if k == len(tails):
                tails.append(x)
            else:
                tails[k] = x
        return len(tails)''',
    "explanation": "Patience sorting. `tails[k]` is the smallest value that can end an increasing subsequence of length `k + 1`. `tails` is always sorted, so the spot for a new number `x` is found by binary search: `bisect_left` returns the first tail that is `>= x`. Replacing that tail with `x` keeps every length's end as small as possible, which leaves more room for later numbers. If `x` is larger than every tail, it extends the longest subsequence by one. `tails` is not always a real subsequence itself, but its length is the answer.\n\nTime O(n log n), space O(n).",
    "explanation_zh": "耐心排序。`tails[k]` 表示能够结束一个长度为 `k + 1` 的递增子序列的最小值。`tails` 始终有序，因此新数字 `x` 的位置可以用二分查找得到：`bisect_left` 返回第一个 `>= x` 的 tail。用 `x` 替换该位置，可以让每个长度的结尾尽量小，给后面的数字留出更多空间。如果 `x` 比所有 tail 都大，它就把最长的子序列延长一位。`tails` 本身不一定是原数组的子序列，但它的长度就是答案。\n\n时间 O(n log n)，空间 O(n)。",
}
