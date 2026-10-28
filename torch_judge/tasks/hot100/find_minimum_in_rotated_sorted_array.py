"""Find Minimum in Rotated Sorted Array (LeetCode 153)."""

import random


def _large_case():
    # 5000 distinct values. A linear scan also passes comfortably here, so this checks the
    # search at the size limit rather than timing O(log n) against O(n).
    rng = random.Random(153)
    values = sorted(rng.sample(range(-10**4, 10**4 + 1), 5000))
    k = rng.randrange(5000)
    nums = values[k:] + values[:k]
    return {"args": [nums], "expected": values[0]}


def _random_case(rng):
    # Every rotation point appears, including the unrotated order (rotating n times)
    nums = sorted(rng.sample(range(-20, 21), rng.randint(1, 8)))
    k = rng.randrange(len(nums))
    return {"args": [nums[k:] + nums[:k]]}


TASK = {
    "title": "Find Minimum in Rotated Sorted Array",
    "title_zh": "寻找旋转排序数组中的最小值",
    "difficulty": "Medium",
    "leetcode_id": 153,
    "leetcode_slug": "find-minimum-in-rotated-sorted-array",
    "category": "Binary Search",
    "function_name": "findMin",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "An array of distinct integers was sorted in ascending order and then rotated between 1 and `n` times. Rotating `n` times returns the original order. Return the smallest element.\n\nThe algorithm must run in O(log n) time.\n\n**Example 1:** `nums = [4,6,7,1,2]` → `1`\n\n**Example 2:** `nums = [8,9,2,3,5,6,7]` → `2`\n\n**Constraints:**\n- `1 <= n <= 5000`, where `n` is the length of `nums`\n- `-10^4 <= nums[i] <= 10^4`\n- All values in `nums` are distinct",
    "description_zh": "一个由互不相同的整数组成的数组，原本按升序排列，之后被旋转了 1 到 `n` 次（旋转 `n` 次等于没有旋转）。返回其中的最小元素。\n\n算法的时间复杂度必须是 O(log n)。\n\n**示例 1：** `nums = [4,6,7,1,2]` → `1`\n\n**示例 2：** `nums = [8,9,2,3,5,6,7]` → `2`\n\n**约束：**\n- `1 <= n <= 5000`，其中 `n` 是 `nums` 的长度\n- `-10^4 <= nums[i] <= 10^4`\n- `nums` 中的值互不相同",
    "hint": "Compare `nums[mid]` with `nums[hi]`, not `nums[lo]`. If `nums[mid]` is larger, the minimum is to its right. Otherwise the minimum is at `mid` or to its left.",
    "hint_zh": "把 `nums[mid]` 和 `nums[hi]` 比较，而不是和 `nums[lo]` 比较。如果 `nums[mid]` 更大，最小值在它右边；否则最小值在 `mid` 或它的左边。",
    "cases": [
        {"args": [[4, 6, 7, 1, 2]], "expected": 1},
        {"args": [[8, 9, 2, 3, 5, 6, 7]], "expected": 2},
        {"args": [[1]], "expected": 1},
        {"args": [[2, 1]], "expected": 1},
        # Not rotated (or rotated n times): the minimum is the first element. Comparing mid with
        # nums[0] returns the last element here.
        {"args": [[3, 4, 5]], "expected": 3},
        # Rotated by one: the minimum is the last element
        {"args": [[5, 6, 7, 8, 9, 1]], "expected": 1},
        {"args": [[10, 20, 30, -5, 0, 4]], "expected": -5},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def findMin(self, nums: List[int]) -> int:
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] > nums[hi]:  # the rotation point is right of mid
                lo = mid + 1
            else:  # nums[mid] < nums[hi]: the minimum is at mid or to its left
                hi = mid
        return nums[lo]''',
    "explanation": "The array is two ascending runs, and the minimum is the first element of the second run. Compare `nums[mid]` with `nums[hi]`: if `nums[mid] > nums[hi]`, the drop must be to the right of `mid`, so `mid` cannot be the minimum. Otherwise `mid` is at or before the drop, so keep `mid` as a candidate with `hi = mid`. The loop ends with `lo == hi` at the minimum. Comparing with `nums[hi]` handles the unrotated case without a special check.\n\nTime O(log n), space O(1).",
    "explanation_zh": "数组由两段升序序列组成，最小值是第二段的第一个元素。比较 `nums[mid]` 和 `nums[hi]`：如果 `nums[mid] > nums[hi]`，那个下降点一定在 `mid` 右边，`mid` 不可能是最小值；否则下降点在 `mid` 或它的左边，保留 `mid` 作为候选，令 `hi = mid`。循环结束时 `lo == hi` 即为最小值。和 `nums[hi]` 比较，未旋转的情况也自然正确，无需特判。\n\n时间 O(log n)，空间 O(1)。",
}
