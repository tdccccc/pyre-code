"""Single Number (LeetCode 136)."""

import random


def _large_case():
    # Length 29999 (the largest odd length under the limit): 14999 values twice plus one lone
    # value, placed last. Counting each element with list.count before reaching it is ~9 * 10^8
    # comparisons, several seconds; the XOR pass is linear.
    rng = random.Random(136)
    values = rng.sample(range(-30000, 30001), 15000)
    lone = values[0]
    nums = values[1:] * 2
    rng.shuffle(nums)
    nums.append(lone)
    return {"args": [nums], "expected": lone}


def _random_case(rng):
    k = rng.randint(0, 4)
    values = rng.sample(range(-9, 10), 2 * k + 1)
    nums = [v for v in values[1:] for _ in range(2)] + [values[0]]
    rng.shuffle(nums)
    return {"args": [nums]}


TASK = {
    "title": "Single Number",
    "title_zh": "只出现一次的数字",
    "difficulty": "Easy",
    "leetcode_id": 136,
    "leetcode_slug": "single-number",
    "category": "Techniques",
    "function_name": "singleNumber",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "Every value in the non-empty array `nums` appears exactly twice, except for one value that appears only once. Return that lone value.\n\nYour solution must run in linear time and use constant extra memory.\n\n**Example 1:** `nums = [5,3,9,3,5]` → `9`\n\n**Example 2:** `nums = [-2]` → `-2`\n\n**Constraints:**\n- `1 <= nums.length <= 3 * 10^4`, and `nums.length` is odd\n- `-3 * 10^4 <= nums[i] <= 3 * 10^4`\n- Every value except one appears exactly twice\n\n**Follow-up:** why does your method still work no matter how the pairs are spread through the array?",
    "description_zh": "非空数组 `nums` 中，每个值恰好出现两次，只有一个值例外，它只出现一次。返回这个只出现一次的值。\n\n你的算法需要线性时间，并且只使用常数的额外空间。\n\n**示例 1：** `nums = [5,3,9,3,5]` → `9`\n\n**示例 2：** `nums = [-2]` → `-2`\n\n**约束：**\n- `1 <= nums.length <= 3 * 10^4`，且 `nums.length` 为奇数\n- `-3 * 10^4 <= nums[i] <= 3 * 10^4`\n- 除一个值外，其余每个值都恰好出现两次\n\n**进阶：** 无论成对的元素散布在数组的什么位置，你的做法为什么都成立？",
    "hint": "Think about what XOR does: a value XORed with itself is 0, and a value XORed with 0 is itself.",
    "hint_zh": "想一想异或的性质：一个数与自己异或得 0，与 0 异或得它自己。",
    "cases": [
        {"args": [[5, 3, 9, 3, 5]], "expected": 9},
        {"args": [[-2]], "expected": -2},
        {"args": [[6, 6, -8, 1, 1]], "expected": -8},
        {"args": [[0]], "expected": 0},
        # Pairs are never adjacent: a method that compares neighbours returns 1 here
        {"args": [[1, 0, 1, 0, 2]], "expected": 2},
        {"args": [[30000, -30000, 30000]], "expected": -30000},
        {"args": [[7, 3, 7, 8, 3]], "expected": 8},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        answer = 0
        for x in nums:
            answer ^= x  # equal values cancel out (a ^ a = 0), leaving the lone one
        return answer''',
    "explanation": "XOR has three useful properties: `a ^ a = 0`, `a ^ 0 = a`, and it is commutative and associative. Each value that appears twice cancels itself out, whatever the order of the array, and the lone value survives.\n\nTime O(n), space O(1).",
    "explanation_zh": "异或有三个有用的性质：`a ^ a = 0`，`a ^ 0 = a`，并且满足交换律和结合律。出现两次的值会两两抵消，与数组中的顺序无关，最后剩下只出现一次的那个数。\n\n时间 O(n)，空间 O(1)。",
}
