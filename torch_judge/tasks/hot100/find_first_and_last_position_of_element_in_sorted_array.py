"""Find First and Last Position of Element in Sorted Array (LeetCode 34)."""

import random


def _large_case():
    # 10^5 sorted values, about 10 copies of each. A linear scan also finishes well under 3 s
    # here, so this checks the answer at the size limit rather than timing O(log n) against O(n).
    rng = random.Random(34)
    nums = sorted(rng.randint(0, 10**4) for _ in range(10**5))
    target = 5000
    first = nums.index(target)
    last = len(nums) - 1 - nums[::-1].index(target)
    return {"args": [nums, target], "expected": [first, last]}


def _random_case(rng):
    # The mistake is only wrong when the target occurs twice or more, so take the target from
    # nums most of the time. Few distinct values keep repeats common; sometimes it is absent.
    # Measured: about 9 of 20 cases expose it
    nums = sorted(rng.choices(range(0, 5), k=rng.randint(0, 8)))
    target = rng.choice(nums) if nums and rng.random() < 0.8 else rng.randint(-1, 5)
    return {"args": [nums, target]}


TASK = {
    "title": "Find First and Last Position of Element in Sorted Array",
    "title_zh": "在排序数组中查找元素的第一个和最后一个位置",
    "difficulty": "Medium",
    "leetcode_id": 34,
    "leetcode_slug": "find-first-and-last-position-of-element-in-sorted-array",
    "category": "Binary Search",
    "function_name": "searchRange",
    "params": [("nums", "int[]"), ("target", "int")],
    "returns": "int[]",
    "description_en": "Given an integer array `nums` sorted in non-decreasing order, find where `target` starts and ends. Return `[first, last]` as indices. If `target` does not occur, return `[-1, -1]`.\n\nThe algorithm must run in O(log n) time.\n\n**Example 1:** `nums = [1,2,2,2,5,9], target = 2` → `[1,3]`\n\n**Example 2:** `nums = [3,8,8,11], target = 5` → `[-1,-1]`\n\n**Constraints:**\n- `0 <= nums.length <= 10^5`\n- `-10^9 <= nums[i], target <= 10^9`\n- `nums` is sorted in non-decreasing order",
    "description_zh": "给定一个按非递减顺序排列的整数数组 `nums`，求 `target` 出现的起始位置和结束位置（下标），以 `[first, last]` 的形式返回。如果 `target` 不存在，返回 `[-1, -1]`。\n\n算法的时间复杂度必须是 O(log n)。\n\n**示例 1：** `nums = [1,2,2,2,5,9], target = 2` → `[1,3]`\n\n**示例 2：** `nums = [3,8,8,11], target = 5` → `[-1,-1]`\n\n**约束：**\n- `0 <= nums.length <= 10^5`\n- `-10^9 <= nums[i], target <= 10^9`\n- `nums` 按非递减顺序排列",
    "hint": "Run two binary searches: one finds the first index with a value `>= target`, the other the first index with a value `> target`. The second one minus 1 is the last position.",
    "hint_zh": "做两次二分：一次找第一个 `>= target` 的下标，另一次找第一个 `> target` 的下标。第二个下标减 1 就是最后的位置。",
    "cases": [
        {"args": [[1, 2, 2, 2, 5, 9], 2], "expected": [1, 3]},
        {"args": [[3, 8, 8, 11], 5], "expected": [-1, -1]},
        # A plain binary search lands on a middle copy (index 2) and returns [2, 2]; both ends
        # have to be searched for
        {"args": [[1, 2, 2, 2, 5, 9], 2], "expected": [1, 3]},
        {"args": [[], 0], "expected": [-1, -1]},
        {"args": [[7], 7], "expected": [0, 0]},
        {"args": [[7], 8], "expected": [-1, -1]},
        # The whole array is one run: first is 0 and last is len - 1
        {"args": [[3, 3, 3], 3], "expected": [0, 2]},
        # Run starting at index 0, and a single copy at the very end
        {"args": [[1, 1, 2, 4, 4], 1], "expected": [0, 1]},
        {"args": [[1, 3, 3, 8], 8], "expected": [3, 3]},
        {"args": [[2, 4, 4, 6], 1], "expected": [-1, -1]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        def lower_bound(x: int) -> int:  # first index with nums[i] >= x
            lo, hi = 0, len(nums)
            while lo < hi:
                mid = (lo + hi) // 2
                if nums[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        first = lower_bound(target)
        if first == len(nums) or nums[first] != target:
            return [-1, -1]
        return [first, lower_bound(target + 1) - 1]''',
    "explanation": "Two lower-bound searches. `lower_bound(x)` is the first index whose value is at least `x`. The first position of `target` is `lower_bound(target)`, after checking that the value there really is `target`. The last position is the index just before `lower_bound(target + 1)`, the first value above `target`. Both searches are the same loop, so there is no separate code for the two ends.\n\nTime O(log n), space O(1).",
    "explanation_zh": "两次下界查找。`lower_bound(x)` 是第一个不小于 `x` 的下标。`target` 的起始位置就是 `lower_bound(target)`，但要先确认那里的值确实等于 `target`。结束位置是 `lower_bound(target + 1)` 的前一个下标，即第一个大于 `target` 的位置减 1。两次查找用的是同一个循环，不需要为两端分别写代码。\n\n时间 O(log n)，空间 O(1)。",
}
