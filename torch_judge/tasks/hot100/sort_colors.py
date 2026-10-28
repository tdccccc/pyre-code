"""Sort Colors (LeetCode 75)."""

import random


def _large_case():
    # n = 300 (the constraint), so no approach can time out here. The case checks that the
    # in-place partition is right at full size.
    rng = random.Random(75)
    nums = [rng.randint(0, 2) for _ in range(300)]
    counts = [nums.count(v) for v in range(3)]
    return {"args": [nums], "expected": [0] * counts[0] + [1] * counts[1] + [2] * counts[2]}


def _random_case(rng):
    return {"args": [[rng.randint(0, 2) for _ in range(rng.randint(1, 10))]]}


TASK = {
    "title": "Sort Colors",
    "title_zh": "颜色分类",
    "difficulty": "Medium",
    "leetcode_id": 75,
    "leetcode_slug": "sort-colors",
    "category": "Techniques",
    "function_name": "sortColors",
    "params": [("nums", "int[]")],
    "returns": "void",
    "inplace": "nums",
    "description_en": "Given an array `nums` that holds only the values `0`, `1` and `2`, rearrange it **in place** so that equal values are grouped in ascending order: all `0`s first, then all `1`s, then all `2`s. Do not return anything, and do not call a built-in sort on the array.\n\n**Example 1:** `nums = [1,2,0,2,1,0]` → `[0,0,1,1,2,2]`\n\n**Example 2:** `nums = [1,2,0]` → `[0,1,2]`\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 300$\n- `nums[i]` is `0`, `1` or `2`\n\n**Follow-up:** can you finish in a single scan of the array, using only three index variables?",
    "description_zh": "给定数组 `nums`，其中只包含 `0`、`1` 和 `2`。请**原地**重新排列，使相同的数聚在一起并按升序排列：先是所有 `0`，然后是所有 `1`，最后是所有 `2`。不要返回任何值，也不要对数组调用内置排序。\n\n**示例 1：** `nums = [1,2,0,2,1,0]` → `[0,0,1,1,2,2]`\n\n**示例 2：** `nums = [1,2,0]` → `[0,1,2]`\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 300$\n- `nums[i]` 为 `0`、`1` 或 `2`\n\n**进阶：** 你能只扫描数组一遍，并且只使用三个下标变量吗？",
    "hint": "Keep three indices: `low` is where the next 0 goes, `high` where the next 2 goes, and `mid` scans. A 2 swapped in from the back is still unexamined, so `mid` must not advance after that swap.",
    "hint_zh": "用三个下标：`low` 是下一个 0 应放的位置，`high` 是下一个 2 应放的位置，`mid` 负责扫描。从后面换过来的值还没有检查，所以与 `high` 交换后 `mid` 不前进。",
    "cases": [
        {"args": [[1, 2, 0, 2, 1, 0]], "expected": [0, 0, 1, 1, 2, 2]},
        {"args": [[1, 2, 0]], "expected": [0, 1, 2]},
        {"args": [[0]], "expected": [0]},
        {"args": [[2]], "expected": [2]},
        {"args": [[1, 1, 1]], "expected": [1, 1, 1]},
        {"args": [[2, 2, 0, 0]], "expected": [0, 0, 2, 2]},
        {"args": [[2, 1, 0, 2, 1]], "expected": [0, 1, 1, 2, 2]},
        # A 2 swapped to the back brings in a 0 that must still be examined
        {"args": [[2, 0]], "expected": [0, 2]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def sortColors(self, nums: List[int]) -> None:
        # Dutch national flag: [0, low) are 0s, [low, mid) are 1s, (high, n) are 2s
        low, mid, high = 0, 0, len(nums) - 1
        while mid <= high:
            if nums[mid] == 0:
                nums[low], nums[mid] = nums[mid], nums[low]
                low += 1
                mid += 1
            elif nums[mid] == 1:
                mid += 1
            else:
                # The value brought in from the back is unexamined, so mid stays put
                nums[mid], nums[high] = nums[high], nums[mid]
                high -= 1''',
    "explanation": "Dutch national flag partition with three indices. Everything before `low` is a 0, everything after `high` is a 2, and `mid..high` is not yet examined. At `mid`, a 0 is swapped to `low`; the value there was a 1 (or `low` equals `mid`), so both pointers advance. A 1 is left alone. A 2 is swapped with `high`, and the value that comes in has not been examined yet, so `mid` does not move.\n\nTime O(n), space O(1).",
    "explanation_zh": "用三个下标做荷兰国旗划分。`low` 之前全是 0，`high` 之后全是 2，`mid` 到 `high` 之间是尚未检查的部分。`mid` 处是 0 时与 `low` 交换：`low` 处原本是 1（或 `low` 与 `mid` 相等），所以两个下标都前进。是 1 时不动。是 2 时与 `high` 交换，换来的值还没检查，所以 `mid` 保持不动。\n\n时间 O(n)，空间 O(1)。",
}
