"""Find the Duplicate Number (LeetCode 287)."""

import random
from collections import Counter


def _large_case():
    # n = 100 000. The repeated value appears three times, so a sum-based formula is wrong here
    rng = random.Random(287)
    n = 100000
    dup = rng.randint(1, n)
    missing = rng.choice([v for v in range(1, n + 1) if v != dup])
    nums = [v for v in range(1, n + 1) if v != missing] + [dup, dup]
    rng.shuffle(nums)
    expected = next(v for v, count in Counter(nums).items() if count > 1)
    return {"args": [nums], "expected": expected}


def _random_case(rng):
    # Build the array from a chosen repeated value: exactly two or three copies, the rest distinct
    n = rng.randint(1, 8)
    dup = rng.randint(1, n)
    count = rng.choice([2, 3]) if n >= 2 else 2
    others = rng.sample([v for v in range(1, n + 1) if v != dup], n + 1 - count)
    nums = [dup] * count + others
    rng.shuffle(nums)
    return {"args": [nums]}


TASK = {
    "title": "Find the Duplicate Number",
    "title_zh": "寻找重复数",
    "difficulty": "Medium",
    "leetcode_id": 287,
    "leetcode_slug": "find-the-duplicate-number",
    "category": "Techniques",
    "function_name": "findDuplicate",
    "params": [("nums", "int[]")],
    "returns": "int",
    "description_en": "You are given an integer array `nums` of $\\texttt{n} + 1$ values, where every value is an integer from `1` to `n`. Exactly one value appears more than once; it may appear two, three or more times. Return that repeated value.\n\nSolve it without modifying `nums`, and using only constant extra memory.\n\n**Example 1:** `nums = [3,1,4,2,3]` → `3`\n\n**Example 2:** `nums = [2,5,5,5,1,3]` → `5` (the repeated value appears three times)\n\n**Constraints:**\n- $1 \\le \\texttt{n} \\le 10^{5}$, so `nums.length = n + 1`\n- $1 \\le \\texttt{nums[i]} \\le \\texttt{n}$\n- Exactly one value repeats, possibly more than twice",
    "description_zh": "给定整数数组 `nums`，包含 $\\texttt{n} + 1$ 个值，每个值都是 `1` 到 `n` 之间的整数。恰好有一个值重复出现，它可能出现两次、三次或更多次。返回这个重复的值。\n\n请在不修改 `nums` 的前提下求解，并且只使用常数级的额外空间。\n\n**示例 1：** `nums = [3,1,4,2,3]` → `3`\n\n**示例 2：** `nums = [2,5,5,5,1,3]` → `5`（重复的值出现了三次）\n\n**约束：**\n- $1 \\le \\texttt{n} \\le 10^{5}$，因此 `nums.length = n + 1`\n- $1 \\le \\texttt{nums[i]} \\le \\texttt{n}$\n- 恰好有一个值重复，且可能出现两次以上",
    "hint": "Read the array as a linked list where index i points to index nums[i]. The values lie in 1..n while there are n + 1 indices, so the walk must enter a cycle, and the repeated value is where the cycle starts.",
    "hint_zh": "把数组看成链表：下标 i 指向下标 nums[i]。由于值在 1 到 n 之间而下标有 n + 1 个，从 0 出发的路径一定会进入一个环，而重复值正是环的入口。",
    "cases": [
        {"args": [[3, 1, 4, 2, 3]], "expected": 3},
        {"args": [[2, 5, 5, 5, 1, 3]], "expected": 5},
        {"args": [[1, 1]], "expected": 1},
        # Three copies of 2: the sum formula (sum - 1 - 2 - ... - n) returns 3 here
        {"args": [[2, 2, 2]], "expected": 2},
        {"args": [[1, 4, 2, 3, 4]], "expected": 4},
        {"args": [[4, 1, 2, 3, 1, 1]], "expected": 1},
        # The repeated value sits at both ends of the array
        {"args": [[5, 4, 3, 2, 1, 5]], "expected": 5},
        {"args": [[6, 6, 1, 2, 3, 4, 5]], "expected": 6},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def findDuplicate(self, nums: List[int]) -> int:
        # Floyd's cycle detection on the functional graph i -> nums[i], from index 0
        slow = fast = nums[0]
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast:
                break
        # Restart one pointer from the beginning: they meet at the cycle's entrance
        slow = nums[0]
        while slow != fast:
            slow = nums[slow]
            fast = nums[fast]
        return slow''',
    "explanation": "Read the array as a linked list: index `i` points to index `nums[i]`. Index 0 has no incoming edge, and the values are at most `n`, so every walk from index 0 ends in a cycle. Two indices with the same value both point to it, so the repeated value is exactly where the cycle begins. Floyd's method finds it: a slow pointer moves one step and a fast pointer two steps until they meet inside the cycle. Restarting one pointer at index 0 and moving both one step at a time, they meet again at the entrance, which is the repeated value. The array is only read, never changed.\n\nTime O(n), space O(1).",
    "explanation_zh": "把数组看成链表：下标 `i` 指向下标 `nums[i]`。下标 0 没有入边，而且所有值都不超过 `n`，所以从 0 出发的路径必定进入环。两个值相同的下标都指向同一个位置，因此重复值正是环的入口。Floyd 算法可以找到它：慢指针每次走一步、快指针每次走两步，直到在环内相遇。然后把一个指针放回下标 0，两个指针都每次走一步，它们会在入口处再次相遇，相遇位置就是重复值。整个过程只读取数组，不修改它。\n\n时间 O(n)，空间 O(1)。",
}
