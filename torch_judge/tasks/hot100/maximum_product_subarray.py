"""Maximum Product Subarray (LeetCode 152)."""

import random


def _max_product(nums: list[int]) -> int:
    # A different method from the reference: split at zeros; within a zero-free block, the best
    # product is a prefix or a suffix of it (the excluded end can only shrink |product|, and
    # it cannot be 0). Scan every prefix and suffix of each block
    best = max(nums)
    blocks, cur = [], []
    for x in nums:
        if x == 0:
            blocks.append(cur)
            cur = []
            best = max(best, 0)
        else:
            cur.append(x)
    blocks.append(cur)
    for block in blocks:
        for seq in (block, block[::-1]):
            prod = 1
            for x in seq:
                prod *= x
                best = max(best, prod)
    return best


def _large_case():
    # Mostly ±1, with zeros cutting the array into short stretches and a few larger magnitudes.
    # Long stretches of big values would overflow 32 bits, so the generator keeps the first
    # seed whose answer fits (the statement promises that)
    rng = random.Random(152)
    while True:
        nums = []
        for _ in range(20000):
            x = rng.random()
            if x < 0.1:
                nums.append(0)
            elif x < 0.12:
                nums.append(rng.choice([-1, 1]) * rng.randint(2, 10))
            else:
                nums.append(rng.choice([-1, 1]))
        expected = _max_product(nums)
        if abs(expected) < 2**31:
            return {"args": [nums], "expected": expected}


def _random_case(rng):
    # Small values in both signs, so a negative pair (or a lone negative) decides the answer often
    return {"args": [[rng.randint(-3, 3) for _ in range(rng.randint(1, 8))]]}


TASK = {
    "title": "Maximum Product Subarray",
    "title_zh": "乘积最大子数组",
    "difficulty": "Medium",
    "leetcode_id": 152,
    "leetcode_slug": "maximum-product-subarray",
    "category": "Dynamic Programming",
    "function_name": "maxProduct",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "Given an integer array `nums`, find the contiguous, non-empty subarray whose product is largest, and return that product.\n\nThe answer is guaranteed to fit in a 32-bit signed integer.\n\n**Example 1:** `nums = [-2,5,-3,-1]` → `30` (the subarray `[-2,5,-3]`)\n\n**Example 2:** `nums = [0,-4,-2,3]` → `24` (the subarray `[-4,-2,3]`)\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 2 \\times 10^{4}$\n- $-10 \\le \\texttt{nums[i]} \\le 10$",
    "description_zh": "给定整数数组 `nums`，找出乘积最大的连续非空子数组，并返回该乘积。\n\n保证答案可以用 32 位有符号整数表示。\n\n**示例 1：** `nums = [-2,5,-3,-1]` → `30`（子数组 `[-2,5,-3]`）\n\n**示例 2：** `nums = [0,-4,-2,3]` → `24`（子数组 `[-4,-2,3]`）\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 2 \\times 10^{4}$\n- $-10 \\le \\texttt{nums[i]} \\le 10$",
    "hint": "A negative number turns the smallest product so far into the largest. Track both the largest and the smallest product of subarrays ending at the current element.",
    "hint_zh": "负数会把此前最小的乘积变成最大的。因此要同时记录以当前元素结尾的最大乘积和最小乘积。",
    "cases": [
        {"args": [[-2, 5, -3, -1]], "expected": 30},
        {"args": [[0, -4, -2, 3]], "expected": 24},
        {"args": [[7]], "expected": 7},
        {"args": [[-3]], "expected": -3},
        {"args": [[0]], "expected": 0},
        # A zero ends every run: -2 and -1 cannot be multiplied across it
        {"args": [[-2, 0, -1]], "expected": 0},
        {"args": [[-1, -2, -3, 0]], "expected": 6},
        # Best subarray is [-2,-4,3]: two negatives make a positive, and the 2 and -5 before
        # them only drag the product down
        {"args": [[2, -5, -2, -4, 3]], "expected": 24},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def maxProduct(self, nums: List[int]) -> int:
        best = cur_max = cur_min = nums[0]
        for x in nums[1:]:
            # Largest and smallest products of subarrays ending at the previous element, times x
            hi = max(x, cur_max * x, cur_min * x)
            lo = min(x, cur_max * x, cur_min * x)
            cur_max, cur_min = hi, lo
            best = max(best, cur_max)
        return best''',
    "explanation": "Keep both the largest and the smallest product of subarrays that end at the current element. Multiplying by a negative number turns the smallest product into the largest one, so the minimum has to be tracked too. For each `x`, the new maximum is `max(x, max·x, min·x)` and the new minimum is `min(x, max·x, min·x)`, where `max` and `min` refer to the previous element. The answer is the largest maximum seen.\n\nTime O(n), space O(1).",
    "explanation_zh": "同时记录以当前元素结尾的子数组的最大乘积和最小乘积。乘以负数会把最小乘积变成最大乘积，所以最小值也必须跟踪。对于每个 `x`，新的最大值为 `max(x, max·x, min·x)`，新的最小值为 `min(x, max·x, min·x)`，其中 `max` 和 `min` 指前一个元素结尾时的值。答案是出现过的最大的最大值。\n\n时间 O(n)，空间 O(1)。",
}
