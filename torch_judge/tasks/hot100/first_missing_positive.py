"""First Missing Positive (LeetCode 41)."""

import random


def _large_case():
    # 5·10^5 values: 1..n-1 shuffled, plus one junk value, so the answer is n. Checking
    # "is k in the list?" for k = 1, 2, ... scans most of the list each time: about n² steps
    rng = random.Random(41)
    n = 500000
    nums = list(range(1, n))
    nums.append(rng.choice([-7, 0, n + 1, 10**9]))
    rng.shuffle(nums)
    present = set(nums)
    expected = 1
    while expected in present:
        expected += 1
    return {"args": [nums], "expected": expected}


def _random_case(rng):
    # Values above the length, zeros, negatives and duplicates all show up here
    return {"args": [[rng.randint(-3, 8) for _ in range(rng.randint(1, 6))]]}


TASK = {
    "title": "First Missing Positive",
    "title_zh": "缺失的第一个正数",
    "difficulty": "Hard",
    "leetcode_id": 41,
    "leetcode_slug": "first-missing-positive",
    "category": "Arrays",
    "function_name": "firstMissingPositive",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "Given an unsorted integer array `nums`, return the smallest positive integer that does not appear in `nums`.\n\n**Example 1:** `nums = [7,1,-4,2]` → `3`\n\n**Example 2:** `nums = [2,2,5]` → `1`\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 5 \\times 10^{5}$\n- $-2^{31} \\le \\texttt{nums[i]} \\le 2^{31} - 1$\n\n**Follow-up:** can you run in O(n) time with O(1) extra space? You may rearrange `nums` while you work.",
    "description_zh": "给定一个未排序的整数数组 `nums`，返回未出现在其中的最小正整数。\n\n**示例 1：** `nums = [7,1,-4,2]` → `3`\n\n**示例 2：** `nums = [2,2,5]` → `1`\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 5 \\times 10^{5}$\n- $-2^{31} \\le \\texttt{nums[i]} \\le 2^{31} - 1$\n\n**进阶：** 你能在 O(n) 时间内、只使用 O(1) 的额外空间完成吗？（过程中可以重新排列 `nums`。）",
    "hint": "The answer always lies between 1 and n + 1, so values outside 1..n can be ignored. Try moving each value v in 1..n to index v - 1 by swapping.",
    "hint_zh": "答案一定在 1 到 n + 1 之间，所以超出 1..n 范围的值可以忽略。试着通过交换，把每个值 v（1 ≤ v ≤ n）放到下标 v - 1 处。",
    "cases": [
        {"args": [[7, 1, -4, 2]], "expected": 3},
        {"args": [[2, 2, 5]], "expected": 1},
        {"args": [[1, 2, 3]], "expected": 4},
        {"args": [[1]], "expected": 2},
        {"args": [[0]], "expected": 1},
        # Every value is above n: the swap logic must skip them instead of indexing past the end
        {"args": [[9, 8, 7]], "expected": 1},
        # A duplicate must not be swapped forever
        {"args": [[1, 1]], "expected": 2},
        {"args": [[4, -2, 6, 3, 1]], "expected": 2},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def firstMissingPositive(self, nums: List[int]) -> int:
        n = len(nums)
        # Put each value v in 1..n at index v - 1; anything else is left where it is
        for i in range(n):
            while 1 <= nums[i] <= n and nums[nums[i] - 1] != nums[i]:
                j = nums[i] - 1
                nums[i], nums[j] = nums[j], nums[i]
        for i in range(n):
            if nums[i] != i + 1:
                return i + 1
        return n + 1''',
    "explanation": "Cyclic sort. The answer is always in 1..n+1, so only values in 1..n matter. Each such value is swapped into its own index (`v` goes to index `v - 1`) until it sits there or the slot already holds the same value, which handles duplicates. Each swap puts one more value in its final place, so there are at most n swaps. The first index whose value is not `index + 1` gives the answer; if every slot is correct, the answer is n + 1.\n\nTime O(n), space O(1).",
    "explanation_zh": "循环排序。答案一定在 1..n+1 之间，所以只有 1..n 范围内的值有用。每个这样的值 `v` 都通过交换放到下标 `v - 1` 处，直到它就位，或者目标位置已经是相同的值（这样就能处理重复值）。每次交换都让一个值就位，所以交换次数不超过 n。第一个值不等于 `下标 + 1` 的位置就是答案；如果所有位置都正确，答案是 n + 1。\n\n时间 O(n)，空间 O(1)。",
}
