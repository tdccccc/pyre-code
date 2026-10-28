"""Product of Array Except Self (LeetCode 238)."""

import random


def _product_except_self(nums):
    # Two arrays: products of everything to the left of i, and everything to the right of i
    n = len(nums)
    left = [1] * n
    for i in range(1, n):
        left[i] = left[i - 1] * nums[i - 1]
    right = [1] * n
    for i in range(n - 2, -1, -1):
        right[i] = right[i + 1] * nums[i + 1]
    return [a * b for a, b in zip(left, right)]


def _large_case():
    # 10^5 values: mostly ±1, with 18 values of ±2 or ±3. Every answer stays below 2^31 (the
    # largest possible product is 3^18), but the numbers still vary. A nested loop needs 10^10
    # multiplications here
    rng = random.Random(238)
    n = 100000
    nums = [rng.choice((-1, 1)) for _ in range(n)]
    for i in rng.sample(range(n), 18):
        nums[i] = rng.choice((-3, -2, 2, 3))
    return {"args": [nums], "expected": _product_except_self(nums)}


def _random_case(rng):
    return {"args": [[rng.randint(-3, 3) for _ in range(rng.randint(2, 6))]]}


TASK = {
    "title": "Product of Array Except Self",
    "title_zh": "除自身以外数组的乘积",
    "difficulty": "Medium",
    "leetcode_id": 238,
    "leetcode_slug": "product-of-array-except-self",
    "category": "Arrays",
    "function_name": "productExceptSelf",
    "params": [("nums", "int[]")],
    "returns": "int[]",
    "description_en": "Given an integer array `nums`, return an array `answer` where `answer[i]` is the product of every element of `nums` except `nums[i]`.\n\nDo not use division.\n\n**Example 1:** `nums = [2,3,4,5]` → `[60,40,30,24]`\n\n**Example 2:** `nums = [-1,0,4]` → `[0,-4,0]`\n\n**Constraints:**\n- $2 \\le \\texttt{nums.length} \\le 10^{5}$\n- $-30 \\le \\texttt{nums[i]} \\le 30$\n- Every answer fits in a 32-bit signed integer\n\n**Follow-up:** can you do it in O(n) time with O(1) extra space, not counting the output array?",
    "description_zh": "给定整数数组 `nums`，返回数组 `answer`，其中 `answer[i]` 等于 `nums` 中除 `nums[i]` 以外所有元素的乘积。\n\n不能使用除法。\n\n**示例 1：** `nums = [2,3,4,5]` → `[60,40,30,24]`\n\n**示例 2：** `nums = [-1,0,4]` → `[0,-4,0]`\n\n**约束：**\n- $2 \\le \\texttt{nums.length} \\le 10^{5}$\n- $-30 \\le \\texttt{nums[i]} \\le 30$\n- 每个答案都能用 32 位有符号整数表示\n\n**进阶：** 你能在 O(n) 时间内完成，且除输出数组外只使用 O(1) 的额外空间吗？",
    "hint": "Build the answer in two passes. The first pass multiplies in everything to the left of each index; the second pass multiplies in everything to the right.",
    "hint_zh": "分两遍计算。第一遍乘上每个下标左边所有元素的积；第二遍乘上右边所有元素的积。",
    "cases": [
        {"args": [[2, 3, 4, 5]], "expected": [60, 40, 30, 24]},
        {"args": [[-1, 0, 4]], "expected": [0, -4, 0]},
        {"args": [[5, 6]], "expected": [6, 5]},
        # Two zeros: every answer is 0. Dividing the total product by each element fails here
        {"args": [[0, 0, 7]], "expected": [0, 0, 0]},
        # One zero: only the zero's own position gets a nonzero answer
        {"args": [[0, 3, -2]], "expected": [-6, 0, 0]},
        {"args": [[-2, -3, 1, 1]], "expected": [-3, -2, 6, 6]},
        {"args": [[1, -1, 1, -1]], "expected": [1, -1, 1, -1]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        n = len(nums)
        out = [1] * n
        prefix = 1  # product of everything left of i
        for i in range(n):
            out[i] = prefix
            prefix *= nums[i]
        suffix = 1  # product of everything right of i
        for i in range(n - 1, -1, -1):
            out[i] *= suffix
            suffix *= nums[i]
        return out''',
    "explanation": "Prefix and suffix products. The first pass stores in `out[i]` the product of everything left of `i`. The second pass walks from the right, keeping a running product of everything right of `i`, and multiplies it in. Division is never used, so zeros need no special case.\n\nTime O(n), space O(1) beyond the output array.",
    "explanation_zh": "前缀积和后缀积。第一遍把下标 `i` 左边所有元素的积存入 `out[i]`。第二遍从右往左扫描，用一个不断更新的变量记录 `i` 右边所有元素的积，并乘进 `out[i]`。全程不用除法，所以遇到 0 也不需要特殊处理。\n\n时间 O(n)，除输出数组外空间 O(1)。",
}
