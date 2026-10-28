"""Median of Two Sorted Arrays (LeetCode 4)."""

import random


def _large_case():
    # 1999 values in total. Merging and sorting also finishes well under 3 s here, so this checks
    # the answer at the size limit rather than timing O(log(m + n)) against O(m + n).
    rng = random.Random(4)
    a = sorted(rng.randint(-10**6, 10**6) for _ in range(1000))
    b = sorted(rng.randint(-10**6, 10**6) for _ in range(999))
    merged = sorted(a + b)
    return {"args": [a, b], "expected": float(merged[len(merged) // 2])}


def _random_case(rng):
    # Either array may be empty; duplicates across the two arrays are common
    while True:
        a = sorted(rng.randint(-6, 6) for _ in range(rng.randint(0, 4)))
        b = sorted(rng.randint(-6, 6) for _ in range(rng.randint(0, 4)))
        if a or b:
            return {"args": [a, b]}


TASK = {
    "title": "Median of Two Sorted Arrays",
    "title_zh": "寻找两个正序数组的中位数",
    "difficulty": "Hard",
    "leetcode_id": 4,
    "leetcode_slug": "median-of-two-sorted-arrays",
    "category": "Binary Search",
    "function_name": "findMedianSortedArrays",
    "params": [("nums1", "int[]"), ("nums2", "int[]")],
    "returns": "float",
    "check": "float",
    "description_en": "Given two sorted arrays `nums1` and `nums2`, return the median of all their elements together. The median is the middle value when the total count is odd, or the average of the two middle values when it is even. Answers within 10^-5 of the exact value are accepted.\n\n**Example 1:** `nums1 = [1,4,9], nums2 = [2,3,8]` → `3.5` (merged: 1, 2, 3, 4, 8, 9; the middle pair is 3 and 4)\n\n**Example 2:** `nums1 = [2,7], nums2 = []` → `4.5`\n\n**Constraints:**\n- $0 \\le \\texttt{m},\\ \\texttt{n} \\le 1000$, where `m` and `n` are the lengths of the arrays\n- $1 \\le \\texttt{m} + \\texttt{n} \\le 2000$\n- $-10^{6} \\le \\texttt{nums1[i]},\\ \\texttt{nums2[i]} \\le 10^{6}$\n- Both arrays are sorted in ascending order\n\n**Follow-up:** can you solve it in O(log(m + n)) time?",
    "description_zh": "给定两个升序排列的数组 `nums1` 和 `nums2`，返回它们所有元素合在一起的中位数。总个数为奇数时，中位数是中间的那个值；为偶数时，是中间两个值的平均数。与精确值相差不超过 10^-5 的答案均可接受。\n\n**示例 1：** `nums1 = [1,4,9], nums2 = [2,3,8]` → `3.5`（合并后为 1, 2, 3, 4, 8, 9，中间两个数是 3 和 4）\n\n**示例 2：** `nums1 = [2,7], nums2 = []` → `4.5`\n\n**约束：**\n- $0 \\le \\texttt{m},\\ \\texttt{n} \\le 1000$，其中 `m` 和 `n` 分别是两个数组的长度\n- $1 \\le \\texttt{m} + \\texttt{n} \\le 2000$\n- $-10^{6} \\le \\texttt{nums1[i]},\\ \\texttt{nums2[i]} \\le 10^{6}$\n- 两个数组都按升序排列\n\n**进阶：** 你能设计一个时间复杂度为 O(log(m + n)) 的算法吗？",
    "hint": "Binary search for how many elements to take from the shorter array, so that the left part holds half of all elements and every left value is at most every right value.",
    "hint_zh": "对较短的数组二分查找：从它取多少个元素进入左半部分，使左半部分恰好占总数的一半，并且左边的每个值都不大于右边的每个值。",
    "cases": [
        {"args": [[1, 4, 9], [2, 3, 8]], "expected": 3.5},
        {"args": [[2, 7], []], "expected": 4.5},
        # One array empty: the search must not index into it
        {"args": [[], [1, 2, 3, 4]], "expected": 2.5},
        {"args": [[1], [2]], "expected": 1.5},
        # No interleaving: every value of one array is below the other. The split must
        # land exactly on the array boundary
        {"args": [[4, 5, 6], [1, 2, 3]], "expected": 3.5},
        # Odd total: the median is a single value, from the first array here
        {"args": [[1, 2, 3, 4], [7, 8, 9]], "expected": 4.0},
        {"args": [[-5, -3, -1], [0, 7]], "expected": -1.0},
        # Duplicates across the arrays
        {"args": [[2, 2], [2]], "expected": 2.0},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        if len(nums1) > len(nums2):
            nums1, nums2 = nums2, nums1  # binary search the shorter array
        m, n = len(nums1), len(nums2)
        half = (m + n + 1) // 2  # size of the left part
        lo, hi = 0, m
        while lo <= hi:
            i = (lo + hi) // 2  # values taken from nums1 into the left part
            j = half - i  # values taken from nums2
            left1 = nums1[i - 1] if i > 0 else float("-inf")
            right1 = nums1[i] if i < m else float("inf")
            left2 = nums2[j - 1] if j > 0 else float("-inf")
            right2 = nums2[j] if j < n else float("inf")
            if left1 <= right2 and left2 <= right1:  # the split is valid
                if (m + n) % 2:
                    return float(max(left1, left2))
                return (max(left1, left2) + min(right1, right2)) / 2
            if left1 > right2:
                hi = i - 1
            else:
                lo = i + 1
        raise ValueError("inputs must be sorted")''',
    "explanation": "Split both arrays so the left parts together hold `(m + n + 1) // 2` values, and every left value is at most every right value. The split is found by binary searching `i`, how many values `nums1` contributes to the left part, on the shorter array, which keeps the search range small. `j = half - i` is then fixed. If `nums1`'s left part is too large (`left1 > right2`), move `i` left; otherwise move it right. Once the split is valid, an odd total's median is the largest left value, and an even total's median is the average of the largest left value and the smallest right value. Missing sides are padded with infinities.\n\nTime O(log(min(m, n))), space O(1).",
    "explanation_zh": "把两个数组各切一刀，使左半部分共有 `(m + n + 1) // 2` 个值，并且左边的每个值都不大于右边的每个值。切分点通过对较短的数组二分查找 `i` 得到，`i` 表示 `nums1` 贡献到左半部分的个数，这样搜索范围较小。随后 `j = half - i` 也随之确定。如果 `nums1` 的左半部分太大（`left1 > right2`），就把 `i` 往左移，否则往右移。切分合法后，总数为奇数时中位数是左半部分的最大值；为偶数时，是左半部分最大值与右半部分最小值的平均数。缺少的一侧用无穷大补齐。\n\n时间 O(log(min(m, n)))，空间 O(1)。",
}
