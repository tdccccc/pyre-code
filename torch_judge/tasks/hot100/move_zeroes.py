"""Move Zeroes (LeetCode 283)."""

import random


def _large_case():
    rng = random.Random(283)
    nums = [rng.choice([0, 0, rng.randint(-1000, 1000)]) for _ in range(100000)]
    nonzero = [x for x in nums if x != 0]
    return {"args": [nums], "expected": nonzero + [0] * (len(nums) - len(nonzero))}


TASK = {
    "title": "Move Zeroes",
    "title_zh": "移动零",
    "difficulty": "Easy",
    "leetcode_id": 283,
    "leetcode_slug": "move-zeroes",
    "category": "Two Pointers",
    "function_name": "moveZeroes",
    "params": [("nums", "int[]")],
    "returns": "void",
    "inplace": "nums",
    "description_en": "Given an integer array `nums`, move every `0` to the end while keeping the non-zero elements in their original relative order.\n\nModify `nums` **in place**; do not return anything and do not build a copy of the array.\n\n**Example 1:** `nums = [0,3,0,7,9]` → `nums` becomes `[3,7,9,0,0]`\n\n**Example 2:** `nums = [0]` → `[0]`\n\n**Constraints:**\n- `1 <= nums.length <= 10^4`\n- `-2^31 <= nums[i] <= 2^31 - 1`\n\n**Follow-up:** can you keep the number of writes small?",
    "description_zh": "给定整数数组 `nums`，把所有 `0` 移到末尾，同时保持非零元素原有的相对顺序。\n\n必须**原地**修改 `nums`，不要返回任何值，也不要复制数组。\n\n**示例 1：** `nums = [0,3,0,7,9]` → `nums` 变为 `[3,7,9,0,0]`\n\n**示例 2：** `nums = [0]` → `[0]`\n\n**约束：**\n- `1 <= nums.length <= 10^4`\n- `-2^31 <= nums[i] <= 2^31 - 1`\n\n**进阶：** 你能尽量减少写操作的次数吗？",
    "hint": "Keep a write pointer for where the next non-zero goes. Scan with a read pointer; when it sees a non-zero, swap it into the write position.",
    "hint_zh": "维护一个「下一个非零元素该放哪」的写指针。读指针向后扫描，遇到非零元素就把它交换到写指针的位置。",
    "cases": [
        {"args": [[0, 3, 0, 7, 9]], "expected": [3, 7, 9, 0, 0]},
        {"args": [[0]], "expected": [0]},
        {"args": [[1, 2, 3]], "expected": [1, 2, 3]},
        {"args": [[0, 0, 1]], "expected": [1, 0, 0]},
        {"args": [[-4, 0, 0, 5, 0, -6]], "expected": [-4, 5, -6, 0, 0, 0]},
    ],
    "large_case": _large_case,
    "solution": '''class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        write = 0
        for read in range(len(nums)):
            if nums[read] != 0:
                nums[write], nums[read] = nums[read], nums[write]
                write += 1''',
    "explanation": "Two pointers. `write` marks where the next non-zero belongs. Scanning with `read`, every non-zero is swapped into `write`, which then advances. Everything before `write` is the non-zeros in order; everything from `write` on is zero.\n\nTime O(n), space O(1).",
    "explanation_zh": "双指针。`write` 指向下一个非零元素应放的位置。`read` 向后扫描，遇到非零元素就和 `write` 位置交换，然后 `write` 前进。`write` 之前是按原顺序排列的非零元素，之后全是零。\n\n时间 O(n)，空间 O(1)。",
}
