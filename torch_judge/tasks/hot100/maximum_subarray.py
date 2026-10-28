"""Maximum Subarray (LeetCode 53)."""

import random


def _max_sum(nums):
    # Prefix sums: the best subarray ends at index j and starts right after the smallest prefix
    # sum seen before j. A different formulation from the Kadane solution
    prefix, lowest, best = 0, 0, None
    for x in nums:
        prefix += x
        best = prefix - lowest if best is None else max(best, prefix - lowest)
        lowest = min(lowest, prefix)
    return best


def _large_case():
    # 10^5 values: trying every subarray is about 5·10^9 additions, far beyond 3 s
    rng = random.Random(53)
    nums = [rng.randint(-10**4, 10**4) for _ in range(100000)]
    return {"args": [nums], "expected": _max_sum(nums)}


def _random_case(rng):
    return {"args": [[rng.randint(-5, 5) for _ in range(rng.randint(1, 8))]]}


TASK = {
    "title": "Maximum Subarray",
    "title_zh": "最大子数组和",
    "difficulty": "Medium",
    "leetcode_id": 53,
    "leetcode_slug": "maximum-subarray",
    "category": "Arrays",
    "function_name": "maxSubArray",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "Given an integer array `nums`, find the contiguous non-empty part of the array (a subarray) whose sum is largest, and return that sum.\n\n**Example 1:** `nums = [2,-5,3,4,-1]` → `7` (the subarray `[3,4]`)\n\n**Example 2:** `nums = [-3,-1,-2]` → `-1` (every value is negative, so the best choice is the single largest element)\n\n**Constraints:**\n- `1 <= nums.length <= 10^5`\n- `-10^4 <= nums[i] <= 10^4`\n\n**Follow-up:** can you find the answer in a single pass?",
    "description_zh": "给定整数数组 `nums`，找出和最大的连续非空子数组，返回它的和。\n\n**示例 1：** `nums = [2,-5,3,4,-1]` → `7`（子数组 `[3,4]`）\n\n**示例 2：** `nums = [-3,-1,-2]` → `-1`（所有数都是负数，最优解就是取最大的那个元素）\n\n**约束：**\n- `1 <= nums.length <= 10^5`\n- `-10^4 <= nums[i] <= 10^4`\n\n**进阶：** 你能只扫描一遍就求出答案吗？",
    "hint": "Scan left to right and keep the best sum of a subarray ending at the current element: either start fresh here or extend the previous run.",
    "hint_zh": "从左到右扫描，记录以当前元素结尾的子数组的最大和：要么从当前元素重新开始，要么接在前面的子数组后面。",
    "cases": [
        {"args": [[2, -5, 3, 4, -1]], "expected": 7},
        {"args": [[-3, -1, -2]], "expected": -1},
        {"args": [[5]], "expected": 5},
        # Starting the best sum at 0 instead of the first element returns 0 for any all-negative input
        {"args": [[-7]], "expected": -7},
        {"args": [[0, 0, -1, 0]], "expected": 0},
        {"args": [[8, -19, 5, -4, 20]], "expected": 21},
        {"args": [[1, -2, 3, -2, 5]], "expected": 6},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def maxSubArray(self, nums: List[int]) -> int:
        cur = best = nums[0]  # cur: best sum of a subarray ending at the current element
        for x in nums[1:]:
            cur = max(x, cur + x)
            best = max(best, cur)
        return best''',
    "explanation": "Kadane's algorithm. `cur` is the best sum of a subarray that ends at the current element: either the element alone, or the element added to the best subarray that ended just before it. `best` keeps the largest `cur` seen. Starting from the first element (not 0) keeps all-negative arrays correct.\n\nTime O(n), space O(1).",
    "explanation_zh": "Kadane 算法。`cur` 表示以当前元素结尾的子数组的最大和：要么只取当前元素，要么把它接在前面以某处结尾的最优子数组后面。`best` 记录所有位置上 `cur` 的最大值。从第一个元素（而不是 0）开始，全是负数的数组也能得到正确结果。\n\n时间 O(n)，空间 O(1)。",
}
