"""Rotate Array (LeetCode 189)."""

import random


def _rotated(nums, k):
    # Slicing: the last k % n elements move to the front
    n = len(nums)
    s = k % n
    return nums[n - s:] + nums[:n - s]


def _large_case():
    # 10^5 elements: shifting one step at a time would take k passes, about 10^10 moves
    rng = random.Random(189)
    nums = [rng.randint(-2**31, 2**31 - 1) for _ in range(100000)]
    k = rng.randint(0, 10**5)
    return {"args": [nums, k], "expected": _rotated(nums, k)}


def _random_case(rng):
    # k often runs past the length, which is where a missing "k % n" shows up
    return {"args": [[rng.randint(-9, 9) for _ in range(rng.randint(1, 7))], rng.randint(0, 12)]}


TASK = {
    "title": "Rotate Array",
    "title_zh": "轮转数组",
    "difficulty": "Medium",
    "leetcode_id": 189,
    "leetcode_slug": "rotate-array",
    "category": "Arrays",
    "function_name": "rotate",
    "params": [("nums", "int[]"), ("k", "int")],
    "returns": "void",
    "inplace": "nums",
    "description_en": "Given an integer array `nums` and a non-negative integer `k`, rotate the array to the right by `k` steps: every element moves `k` positions towards the end, and the elements that fall off the end wrap around to the front.\n\nModify `nums` **in place**; do not return anything.\n\n**Example 1:** `nums = [10,20,30,40], k = 1` → `[40,10,20,30]`\n\n**Example 2:** `nums = [1,2,3,4,5], k = 7` → `[4,5,1,2,3]` (7 steps on a length-5 array is the same as 2 steps)\n\n**Constraints:**\n- `1 <= nums.length <= 10^5`\n- `-2^31 <= nums[i] <= 2^31 - 1`\n- `0 <= k <= 10^5`\n\n**Follow-up:** can you do it with O(1) extra space?",
    "description_zh": "给定整数数组 `nums` 和非负整数 `k`，把数组向右轮转 `k` 步：每个元素都向后移动 `k` 个位置，从末尾移出的元素绕回到开头。\n\n必须**原地**修改 `nums`，不要返回任何值。\n\n**示例 1：** `nums = [10,20,30,40], k = 1` → `[40,10,20,30]`\n\n**示例 2：** `nums = [1,2,3,4,5], k = 7` → `[4,5,1,2,3]`（长度为 5 的数组轮转 7 步，等价于轮转 2 步）\n\n**约束：**\n- `1 <= nums.length <= 10^5`\n- `-2^31 <= nums[i] <= 2^31 - 1`\n- `0 <= k <= 10^5`\n\n**进阶：** 你能只用 O(1) 的额外空间完成吗？",
    "hint": "Reduce k modulo the length first. Then reversing the whole array, and then reversing its two parts separately, gives the rotation.",
    "hint_zh": "先把 k 对数组长度取模。然后整体反转数组，再分别反转前 k 个元素和剩下的部分，就得到轮转的结果。",
    "cases": [
        {"args": [[10, 20, 30, 40], 1], "expected": [40, 10, 20, 30]},
        {"args": [[1, 2, 3, 4, 5], 7], "expected": [4, 5, 1, 2, 3]},
        # k larger than the length without reduction: slicing with -k then leaves the array as it was
        {"args": [[1, 2, 3, 4, 5, 6], 4], "expected": [3, 4, 5, 6, 1, 2]},
        {"args": [[1, 2, 3], 3], "expected": [1, 2, 3]},
        {"args": [[7], 3], "expected": [7]},
        {"args": [[1, 2], 0], "expected": [1, 2]},
        {"args": [[-1, -100, 3, 99], 2], "expected": [3, 99, -1, -100]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def rotate(self, nums: List[int], k: int) -> None:
        n = len(nums)
        k %= n

        def reverse(lo: int, hi: int) -> None:
            while lo < hi:
                nums[lo], nums[hi] = nums[hi], nums[lo]
                lo, hi = lo + 1, hi - 1

        reverse(0, n - 1)  # the whole array: the last k elements are now first, reversed
        reverse(0, k - 1)  # put the first k back in order
        reverse(k, n - 1)  # and the rest''',
    "explanation": "Three reversals. Reversing the whole array brings the last `k` elements to the front, but each block is reversed. Reversing the first `k` elements and then the remaining elements puts both blocks back in their original order. `k %= n` handles `k` larger than the length, and `k = 0` leaves the array unchanged.\n\nTime O(n), space O(1).",
    "explanation_zh": "三次反转。先整体反转数组，最后 `k` 个元素会移到最前面，但每一段内部的顺序都是反的。再分别反转前 `k` 个元素和剩下的元素，两段就恢复了原来的顺序。`k %= n` 处理 `k` 超过长度的情况，`k = 0` 时数组保持不变。\n\n时间 O(n)，空间 O(1)。",
}
