"""Two Sum (LeetCode 1)."""

import random


def _large_case():
    # Every value is ≡ 0 (mod 4) except one ≡ 1 and one ≡ 2, and the target is ≡ 3, so the
    # answer is unique. Both indices sit at the end, so an O(n²) scan has to try every pair.
    rng = random.Random(1)
    n = 20000
    nums = [4 * v for v in rng.sample(range(-10**8, 10**8), n)]
    nums[-2] = 4 * 12345 + 1
    nums[-1] = 4 * 67890 + 2
    return {"args": [nums, nums[-2] + nums[-1]], "expected": [n - 2, n - 1]}


def _random_case(rng):
    # Retry until exactly one pair sums to the target; any other valid pair would be
    # graded wrong against the reference answer
    while True:
        nums = [rng.randint(-10, 10) for _ in range(rng.randint(2, 8))]
        i, j = sorted(rng.sample(range(len(nums)), 2))
        target = nums[i] + nums[j]
        pairs = sum(nums[a] + nums[b] == target for a in range(len(nums)) for b in range(a + 1, len(nums)))
        if pairs == 1:
            return {"args": [nums, target]}


TASK = {
    "title": "Two Sum",
    "title_zh": "两数之和",
    "difficulty": "Easy",
    "leetcode_id": 1,
    "leetcode_slug": "two-sum",
    "category": "Hashing",
    "function_name": "twoSum",
    "params": [("nums", "int[]"), ("target", "int")],
    "returns": "int[]",
    "check": "unordered",
    "description_en": "Given an integer array `nums` and an integer `target`, find the two positions whose values add up to `target` and return their indices.\n\nEach input has exactly one valid pair, and the same position cannot be used twice. The two indices may be returned in either order.\n\n**Example 1:** `nums = [4,11,7,2], target = 9` → `[2,3]` (7 + 2 = 9)\n\n**Example 2:** `nums = [5,5], target = 10` → `[0,1]`\n\n**Constraints:**\n- `2 <= nums.length <= 10^4`\n- `-10^9 <= nums[i], target <= 10^9`\n- Exactly one valid answer exists\n\n**Follow-up:** can you do better than O(n²)?",
    "description_zh": "给定整数数组 `nums` 和整数 `target`，找出数组中和为 `target` 的两个位置，返回它们的下标。\n\n每组输入恰好只有一对答案，同一个位置不能使用两次。两个下标可以按任意顺序返回。\n\n**示例 1：** `nums = [4,11,7,2], target = 9` → `[2,3]`（7 + 2 = 9）\n\n**示例 2：** `nums = [5,5], target = 10` → `[0,1]`\n\n**约束：**\n- `2 <= nums.length <= 10^4`\n- `-10^9 <= nums[i], target <= 10^9`\n- 答案恰好存在一个\n\n**进阶：** 你能想出时间复杂度低于 O(n²) 的算法吗？",
    "hint": "While scanning, ask \"have I already seen `target - x`?\" A dict from value to index answers that in O(1).",
    "hint_zh": "遍历时问自己：「`target - x` 之前出现过吗？」用一个 值 → 下标 的字典可以 O(1) 回答。",
    "cases": [
        {"args": [[4, 11, 7, 2], 9], "expected": [2, 3]},
        {"args": [[5, 5], 10], "expected": [0, 1]},
        {"args": [[-3, 4, 1, 90], -2], "expected": [0, 2]},
        {"args": [[0, 4, 3, 0], 0], "expected": [0, 3]},
        {"args": [[1, 2, 3, 4, 5, 6], 11], "expected": [4, 5]},
        # target - x == x with a single x: storing x before the lookup pairs it with itself
        {"args": [[3, 2, 4], 6], "expected": [1, 2]},
        {"args": [[-1, -2, -3, -4, -5], -8], "expected": [2, 4]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        seen = {}  # value -> index
        for i, x in enumerate(nums):
            if target - x in seen:
                return [seen[target - x], i]
            seen[x] = i
        return []''',
    "explanation": "One pass with a hash map from value to index. For each `x`, if `target - x` was seen earlier, those two indices are the answer; otherwise record `x`. Checking before inserting keeps an element from pairing with itself.\n\nTime O(n), space O(n).",
    "explanation_zh": "一次遍历，用哈希表记录 值 → 下标。对每个 `x`，如果 `target - x` 已经出现过，这两个下标就是答案；否则把 `x` 记下来。先查再存，保证不会和自己配对。\n\n时间 O(n)，空间 O(n)。",
}
