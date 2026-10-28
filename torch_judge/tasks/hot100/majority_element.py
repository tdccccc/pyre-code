"""Majority Element (LeetCode 169)."""

import random
from collections import Counter


def _large_case():
    # n = 50000 (the constraint). The majority value (25001 copies) sits at the end, so a
    # check that counts each value from the front does ~1.25 * 10^9 comparisons before it
    # reaches the answer.
    rng = random.Random(169)
    majority = 424242
    others = []
    while len(others) < 24999:
        v = rng.randint(-10**9, 10**9)
        if v != majority:
            others.append(v)
    nums = others + [majority] * 25001
    return {"args": [nums], "expected": Counter(nums).most_common(1)[0][0]}


def _random_case(rng):
    # Odd n with the bare minimum majority is the case where another value can reach n // 2
    # occurrences before the majority does. Half the cases put the other values first, which
    # makes that tie happen; the rest are shuffled
    n = rng.randint(3, 9)
    majority = rng.randint(-5, 5)
    count = n // 2 + 1 if rng.random() < 0.7 else rng.randint(n // 2 + 1, n)
    other = rng.choice([v for v in range(-5, 6) if v != majority])
    nums = [majority] * count + [other] * (n - count)
    if rng.random() < 0.5:
        rng.shuffle(nums)
    else:
        nums = nums[count:] + nums[:count]
    return {"args": [nums]}


TASK = {
    "title": "Majority Element",
    "title_zh": "多数元素",
    "difficulty": "Easy",
    "leetcode_id": 169,
    "leetcode_slug": "majority-element",
    "category": "Techniques",
    "function_name": "majorityElement",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "Given an array `nums` of length `n`, return its majority element: the value that appears more than `n / 2` times. The input always contains such a value.\n\n**Example 1:** `nums = [4,1,4,4,7]` → `4`\n\n**Example 2:** `nums = [0,6,0]` → `0`\n\n**Constraints:**\n- `1 <= n <= 5 * 10^4`\n- `-10^9 <= nums[i] <= 10^9`\n- A majority element is guaranteed to exist\n\n**Follow-up:** the answer is easy to find after sorting. Can you find it faster than sorting, using only a few variables?",
    "description_zh": "给定长度为 `n` 的数组 `nums`，返回它的多数元素：出现次数超过 `n / 2` 的那个值。输入中一定存在这样的值。\n\n**示例 1：** `nums = [4,1,4,4,7]` → `4`\n\n**示例 2：** `nums = [0,6,0]` → `0`\n\n**约束：**\n- `1 <= n <= 5 * 10^4`\n- `-10^9 <= nums[i] <= 10^9`\n- 保证多数元素存在\n\n**进阶：** 排序之后很容易找到它。你能不排序、只用几个变量，比排序更快地找到吗？",
    "hint": "Boyer-Moore voting: a matching value adds a vote, a different one removes a vote, and the candidate is replaced whenever the count reaches zero. Sorting or counting also works.",
    "hint_zh": "Boyer-Moore 投票：相同的值加一票，不同的值减一票，计数归零时更换候选值。排序或计数也可以。",
    "cases": [
        {"args": [[4, 1, 4, 4, 7]], "expected": 4},
        {"args": [[0, 6, 0]], "expected": 0},
        {"args": [[5]], "expected": 5},
        {"args": [[1000000000, -1000000000, 1000000000]], "expected": 1000000000},
        # 2 reaches n // 2 first, but only 1 is the majority: a threshold of n // 2 returns 2
        {"args": [[2, 2, 1, 1, 1]], "expected": 1},
        {"args": [[7, 7, 7, 1]], "expected": 7},
        {"args": [[-1, -1, 8, -1, 8, -1]], "expected": -1},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        # Boyer-Moore voting: a candidate is replaced only when its vote count drops to zero
        candidate, count = 0, 0
        for x in nums:
            if count == 0:
                candidate = x
            count += 1 if x == candidate else -1
        return candidate''',
    "explanation": "Boyer-Moore majority vote. Keep a candidate and a count: a value equal to the candidate adds a vote, and any other value removes one. When the count reaches zero, the next value becomes the candidate. The majority value takes more than half of the array, so the other values cannot cancel all of its votes; whatever candidate is left at the end is the majority.\n\nTime O(n), space O(1).",
    "explanation_zh": "Boyer-Moore 投票算法。维护一个候选值和一个计数：与候选值相同的数加一票，其他数减一票。计数归零时，下一个数成为候选值。多数元素占据超过一半的位置，其他数的票数无法全部抵消它，所以最后留下的候选值就是多数元素。\n\n时间 O(n)，空间 O(1)。",
}
