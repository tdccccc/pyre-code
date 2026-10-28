"""Next Permutation (LeetCode 31)."""

import random


def _next_permutation(nums: list[int]) -> list[int]:
    # A different method from the reference: locate the pivot and its replacement by search,
    # then sort the tail, instead of reversing it
    out = list(nums)
    ascents = [k for k in range(len(out) - 1) if out[k] < out[k + 1]]
    if not ascents:
        return sorted(out)  # already the last arrangement: wrap around to the first
    i = ascents[-1]  # rightmost ascent: everything after it is non-increasing
    j = max(k for k in range(i + 1, len(out)) if out[k] > out[i])  # smallest value above out[i]
    out[i], out[j] = out[j], out[i]
    return out[:i + 1] + sorted(out[i + 1:])


def _large_case():
    # A random prefix, then a 99 000-value descending tail. The pivot sits near the front,
    # so almost the whole array has to be rearranged
    rng = random.Random(31)
    prefix = [rng.randint(-100, 100) for _ in range(1000)]
    tail = sorted((rng.randint(-100, 100) for _ in range(99000)), reverse=True)
    nums = prefix + tail
    return {"args": [nums], "expected": _next_permutation(nums)}


def _random_case(rng):
    # Small values in a narrow range, so duplicates (which need care) show up often
    return {"args": [[rng.randint(-3, 3) for _ in range(rng.randint(1, 6))]]}


TASK = {
    "title": "Next Permutation",
    "title_zh": "下一个排列",
    "difficulty": "Medium",
    "leetcode_id": 31,
    "leetcode_slug": "next-permutation",
    "category": "Techniques",
    "function_name": "nextPermutation",
    "params": [("nums", "int[]")],
    "returns": "void",
    "inplace": "nums",
    "description_en": "A **permutation** of `nums` is any rearrangement of its values. List all the arrangements in dictionary order; the **next permutation** is the one that comes right after `nums` in that list. If `nums` is already the last arrangement (sorted in descending order), the next one is the first arrangement (sorted in ascending order).\n\nRearrange `nums` **in place** into its next permutation. Do not return anything. Use only constant extra memory.\n\n**Example 1:** `nums = [1,4,2]` → `[2,1,4]`\n\n**Example 2:** `nums = [5,3,2]` → `[2,3,5]` (already the last arrangement, so it wraps around)\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 10^{5}$\n- $-100 \\le \\texttt{nums[i]} \\le 100$",
    "description_zh": "`nums` 的一个**排列**是它的值的任意重新排列。把所有排列按字典序排好，**下一个排列**就是紧跟在 `nums` 之后的那一个。如果 `nums` 已经是最后一个排列（降序排列），那么下一个排列就是第一个排列（升序排列）。\n\n请把 `nums` **原地**改写为它的下一个排列。不要返回任何值，只使用常数级的额外内存。\n\n**示例 1：** `nums = [1,4,2]` → `[2,1,4]`\n\n**示例 2：** `nums = [5,3,2]` → `[2,3,5]`（已经是最后一个排列，所以要回到第一个）\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 10^{5}$\n- $-100 \\le \\texttt{nums[i]} \\le 100$",
    "hint": "Scan from the right for the first value that is smaller than its right neighbour (the pivot). Swap the pivot with the smallest larger value to its right, then reverse everything after the pivot's position.",
    "hint_zh": "从右往左找第一个比右邻居小的数（枢轴）。把枢轴与其右侧比它大的最小值交换，然后把枢轴之后的部分反转。",
    "cases": [
        {"args": [[1, 4, 2]], "expected": [2, 1, 4]},
        {"args": [[5, 3, 2]], "expected": [2, 3, 5]},
        {"args": [[1]], "expected": [1]},
        {"args": [[1, 2]], "expected": [2, 1]},
        # Pivot 2 swaps with 3, then the tail [4, 2, 1] must be reversed to [1, 2, 4]
        {"args": [[2, 4, 3, 1]], "expected": [3, 1, 2, 4]},
        {"args": [[1, 3, 2]], "expected": [2, 1, 3]},
        # Duplicates: the replacement must be the smallest value larger than the pivot (5, not a 1)
        {"args": [[1, 5, 1]], "expected": [5, 1, 1]},
        {"args": [[9, 8, 7, 1]], "expected": [1, 7, 8, 9]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        # Pivot: the rightmost index whose value is smaller than the value after it
        i = len(nums) - 2
        while i >= 0 and nums[i] >= nums[i + 1]:
            i -= 1
        if i >= 0:
            # The tail is non-increasing, so the last value larger than the pivot is the smallest such
            j = len(nums) - 1
            while nums[j] <= nums[i]:
                j -= 1
            nums[i], nums[j] = nums[j], nums[i]
        # The tail was in descending order; reversing it gives the smallest arrangement
        nums[i + 1:] = nums[i + 1:][::-1]''',
    "explanation": "Find the rightmost pivot `i` with `nums[i] < nums[i + 1]`. Everything after it is non-increasing, and that tail is already the largest arrangement of its values. To get the next arrangement, the pivot must grow: swap it with the smallest value to its right that is larger than it (scanning from the right finds it), then the tail is reversed into ascending order, which is the smallest arrangement of the new tail. If there is no pivot, the whole array is descending, so reversing it wraps around to the first arrangement.\n\nTime O(n), space O(1).",
    "explanation_zh": "从右往左找最右边的枢轴 `i`，满足 `nums[i] < nums[i + 1]`。它之后的部分是不增的，而且已经是这些值能组成的最大排列。要得到下一个排列，必须让枢轴变大：把它与右侧比它大的最小值交换（从右往左扫描即可找到），然后把剩下的尾部反转成升序，这就是新尾部的最小排列。如果找不到枢轴，说明整个数组是降序的，整体反转即可回到第一个排列。\n\n时间 O(n)，空间 O(1)。",
}
