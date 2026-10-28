"""Search Insert Position (LeetCode 35)."""

import bisect
import random


def _large_case():
    # 10^4 values, so even a linear scan finishes far under 3 s. This checks the answer at the
    # size limit; the intended O(log n) search is not timed against an O(n) scan here.
    rng = random.Random(35)
    nums = sorted(rng.sample(range(-10**4, 10**4 + 1), 10**4))
    target = rng.randint(-10**4 - 5, 10**4 + 5)
    return {"args": [nums, target], "expected": bisect.bisect_left(nums, target)}


def _random_case(rng):
    # Small arrays, and targets often just outside the range, so both ends get exercised
    nums = sorted(rng.sample(range(-10, 11), rng.randint(1, 8)))
    return {"args": [nums, rng.randint(-11, 11)]}


TASK = {
    "title": "Search Insert Position",
    "title_zh": "搜索插入位置",
    "difficulty": "Easy",
    "leetcode_id": 35,
    "leetcode_slug": "search-insert-position",
    "category": "Binary Search",
    "function_name": "searchInsert",
    "params": [("nums", "int[]"), ("target", "int")],
    "returns": "int",
    "description_en": "You are given a sorted array `nums` of distinct integers and a value `target`. If `target` is in the array, return its index. Otherwise, return the index where it would go so that the array stays sorted.\n\nThe algorithm must run in O(log n) time.\n\n**Example 1:** `nums = [2,4,7,9], target = 7` → `2`\n\n**Example 2:** `nums = [2,4,7,9], target = 1` → `0` (smaller than everything, so it goes in front)\n\n**Constraints:**\n- `1 <= nums.length <= 10^4`\n- `-10^4 <= nums[i], target <= 10^4`\n- `nums` is sorted in ascending order and has no duplicates",
    "description_zh": "给定一个升序排列且元素互不相同的数组 `nums`，以及一个目标值 `target`。如果 `target` 在数组中，返回它的下标；否则返回它应该插入的位置，使数组保持升序。\n\n算法的时间复杂度必须是 O(log n)。\n\n**示例 1：** `nums = [2,4,7,9], target = 7` → `2`\n\n**示例 2：** `nums = [2,4,7,9], target = 1` → `0`（比所有元素都小，插在最前面）\n\n**约束：**\n- `1 <= nums.length <= 10^4`\n- `-10^4 <= nums[i], target <= 10^4`\n- `nums` 按升序排列，且没有重复元素",
    "hint": "Binary search for the first index whose value is `>= target`. That index is the answer whether or not `target` is present.",
    "hint_zh": "用二分查找第一个 `>= target` 的下标。无论 `target` 是否存在，这个下标都是答案。",
    "cases": [
        {"args": [[2, 4, 7, 9], 7], "expected": 2},
        {"args": [[2, 4, 7, 9], 1], "expected": 0},
        # Past the last element: a search with hi = len - 1 stops one index short
        {"args": [[2, 4, 7, 9], 12], "expected": 4},
        # Equal to the last element: must return its index, not len
        {"args": [[2, 4, 7, 9], 9], "expected": 3},
        {"args": [[-10, -4, 0, 6], -11], "expected": 0},
        # One element: found, and inserted after it
        {"args": [[-5], -5], "expected": 0},
        {"args": [[-5], 3], "expected": 1},
        {"args": [[1, 3], 2], "expected": 1},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        lo, hi = 0, len(nums)  # the answer lies in [0, len(nums)]
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo''',
    "explanation": "The answer is the first index whose value is at least `target` (the lower bound). Binary search keeps a half-open range `[lo, hi)` that always contains that index, starting from `[0, len(nums)]` so that an insert past the end is covered. When `nums[mid] < target`, every index up to `mid` is too small, so `lo` moves past `mid`; otherwise `mid` is a candidate and `hi` moves onto it. Found and missing targets take the same path.\n\nTime O(log n), space O(1).",
    "explanation_zh": "答案就是第一个不小于 `target` 的下标（下界）。二分查找维护一个半开区间 `[lo, hi)`，它一定包含这个下标。初始区间是 `[0, len(nums)]`，因此插入到末尾的情况也被覆盖。当 `nums[mid] < target` 时，`mid` 及其左边都太小，`lo` 移到 `mid` 之后；否则 `mid` 是候选，`hi` 移到 `mid`。找到和找不到 `target` 走的是同一条路径。\n\n时间 O(log n)，空间 O(1)。",
}
