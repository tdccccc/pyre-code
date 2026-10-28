"""Search in Rotated Sorted Array (LeetCode 33)."""

import random


def _large_case():
    # 5000 distinct values. A linear scan also passes comfortably here, so this checks the
    # rotation logic at the size limit rather than timing O(log n) against O(n).
    rng = random.Random(33)
    values = sorted(rng.sample(range(-10**4, 10**4 + 1), 5000))
    k = rng.randrange(5000)
    nums = values[k:] + values[:k]
    if rng.random() < 0.6:
        target = nums[rng.randrange(5000)]
    else:
        target = rng.randint(-10**4, 10**4)
    expected = nums.index(target) if target in nums else -1
    return {"args": [nums, target], "expected": expected}


def _random_case(rng):
    # Distinct values, rotated by any amount (0 included). The mistake is a target equal to the
    # first or last value of a sorted half, so most targets are the array's two ends; the rest
    # are random, sometimes absent
    nums = sorted(rng.sample(range(-10, 11), rng.randint(1, 8)))
    k = rng.randrange(len(nums))
    rotated = nums[k:] + nums[:k]
    if rng.random() < 0.85:
        target = rng.choice([rotated[0], rotated[-1]])
    else:
        target = rng.randint(-11, 11)
    return {"args": [rotated, target]}


TASK = {
    "title": "Search in Rotated Sorted Array",
    "title_zh": "搜索旋转排序数组",
    "difficulty": "Medium",
    "leetcode_id": 33,
    "leetcode_slug": "search-in-rotated-sorted-array",
    "category": "Binary Search",
    "function_name": "search",
    "params": [("nums", "int[]"), ("target", "int")],
    "returns": "int",
    "description_en": "An array of distinct integers was sorted in ascending order, then rotated at an unknown pivot. For example, `[1,3,5,6,8]` rotated so that it starts at `6` becomes `[6,8,1,3,5]`. The array may also be left unrotated. Given `nums` and `target`, return the index of `target`, or `-1` if it is not in the array.\n\nThe algorithm must run in O(log n) time.\n\n**Example 1:** `nums = [6,8,1,3,5], target = 1` → `2`\n\n**Example 2:** `nums = [9,2,3,5,7], target = 6` → `-1`\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 5000$\n- $-10^{4} \\le \\texttt{nums[i]},\\ \\texttt{target} \\le 10^{4}$\n- All values in `nums` are distinct",
    "description_zh": "一个由互不相同的整数组成的数组，原本按升序排列，之后在某个未知的位置被旋转。例如 `[1,3,5,6,8]` 从 `6` 开始重新排列，得到 `[6,8,1,3,5]`。数组也可能没有被旋转。给定 `nums` 和 `target`，返回 `target` 的下标；如果不存在，返回 `-1`。\n\n算法的时间复杂度必须是 O(log n)。\n\n**示例 1：** `nums = [6,8,1,3,5], target = 1` → `2`\n\n**示例 2：** `nums = [9,2,3,5,7], target = 6` → `-1`\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 5000$\n- $-10^{4} \\le \\texttt{nums[i]},\\ \\texttt{target} \\le 10^{4}$\n- `nums` 中的值互不相同",
    "hint": "At any midpoint, one of the two halves is sorted. Check whether `target` lies inside that sorted half; if it does, search there, otherwise search the other half.",
    "hint_zh": "在任何中点处，左右两半中至少有一半是有序的。判断 `target` 是否落在有序的那一半里：如果在，就去那一半查找；否则去另一半。",
    "cases": [
        {"args": [[6, 8, 1, 3, 5], 1], "expected": 2},
        {"args": [[9, 2, 3, 5, 7], 6], "expected": -1},
        # Target equals the first element, which sits in the left sorted half
        {"args": [[6, 8, 1, 3, 5], 6], "expected": 0},
        # Target is the last element, in the right sorted half
        {"args": [[6, 8, 1, 3, 5], 5], "expected": 4},
        {"args": [[6, 8, 1, 3, 5], 8], "expected": 1},
        # Not rotated at all: the left half is always sorted
        {"args": [[2, 4, 6, 8], 6], "expected": 2},
        {"args": [[1], 1], "expected": 0},
        {"args": [[5, 1, 3], 3], "expected": 2},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def search(self, nums: List[int], target: int) -> int:
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] == target:
                return mid
            if nums[lo] <= nums[mid]:  # [lo, mid] is sorted
                if nums[lo] <= target < nums[mid]:
                    hi = mid - 1
                else:
                    lo = mid + 1
            else:  # [mid, hi] is sorted
                if nums[mid] < target <= nums[hi]:
                    lo = mid + 1
                else:
                    hi = mid - 1
        return -1''',
    "explanation": "Binary search where each midpoint splits the range into two parts, and at least one part is sorted. If the left part `[lo, mid]` is sorted and `target` lies in its value range `[nums[lo], nums[mid])`, keep the left part; otherwise drop it. If the right part is sorted, the same test applies to `(nums[mid], nums[hi]]`. The lower end of each range is inclusive and the upper end exclusive, so a target equal to `nums[lo]` is still found.\n\nTime O(log n), space O(1).",
    "explanation_zh": "二分查找。每个中点都把当前区间分成两部分，其中至少有一部分是有序的。如果左半部分 `[lo, mid]` 有序，并且 `target` 落在它的取值范围 `[nums[lo], nums[mid])` 内，就保留左半部分，否则舍弃；如果右半部分有序，同样的判断用于 `(nums[mid], nums[hi]]`。每个范围的下界包含、上界不包含，所以等于 `nums[lo]` 的 `target` 不会被漏掉。\n\n时间 O(log n)，空间 O(1)。",
}
