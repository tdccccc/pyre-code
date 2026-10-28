"""3Sum (LeetCode 15)."""

import random
from collections import Counter


def _triplets(nums: list[int]) -> list[list[int]]:
    # Independent method: enumerate distinct values a <= b <= c with a + b + c = 0, each used no
    # more often than it occurs in nums. Pairs of distinct values keep this quick.
    counts = Counter(nums)
    values = sorted(counts)
    out = []
    for i, a in enumerate(values):
        for b in values[i:]:
            c = -a - b
            if c < b or c not in counts:
                continue
            need = Counter((a, b, c))
            if all(counts[v] >= k for v, k in need.items()):
                out.append([a, b, c])
    return out


def _large_case():
    # 3000 numbers, the constraint limit. Each of 1500 distinct values appears twice, so the
    # duplicate handling matters. Checking every triple would be about 4·10⁹ comparisons.
    rng = random.Random(15)
    pool = rng.sample(range(-10**5, 10**5 + 1), 1500)
    nums = pool + pool
    rng.shuffle(nums)
    return {"args": [nums], "expected": _triplets(nums)}


def _random_case(rng):
    # Values in -3..3 repeat often, so duplicate triplets are a real risk
    return {"args": [[rng.randint(-3, 3) for _ in range(rng.randint(3, 8))]]}


TASK = {
    "title": "3Sum",
    "title_zh": "三数之和",
    "difficulty": "Medium",
    "leetcode_id": 15,
    "leetcode_slug": "3sum",
    "category": "Two Pointers",
    "function_name": "threeSum",
    "params": [("nums", "int[]")],
    "returns": "int[][]",
    "check": "unordered_nested",
    "description_en": "Given an integer array `nums`, find every triplet `[nums[i], nums[j], nums[k]]` whose three positions are all different and whose values sum to zero. The same set of values must not appear twice in the answer, whatever order it was found in. The order of the triplets, and of the values inside each triplet, does not matter.\n\n**Example 1:** `nums = [2,-2,0,1,-1]` → `[[-2,0,2],[-1,0,1]]`\n\n**Example 2:** `nums = [4,-4,4,-4]` → `[]` (no three values sum to zero)\n\n**Constraints:**\n- `3 <= nums.length <= 3000`\n- `-10^5 <= nums[i] <= 10^5`",
    "description_zh": "给定整数数组 `nums`，找出所有和为零的三元组 `[nums[i], nums[j], nums[k]]`，三个下标两两不同。答案中同一组数值（不论顺序）只能出现一次。三元组的顺序以及三元组内部元素的顺序均不限。\n\n**示例 1：** `nums = [2,-2,0,1,-1]` → `[[-2,0,2],[-1,0,1]]`\n\n**示例 2：** `nums = [4,-4,4,-4]` → `[]`（任意三个数都凑不出 0）\n\n**约束：**\n- `3 <= nums.length <= 3000`\n- `-10^5 <= nums[i] <= 10^5`",
    "hint": "Sort first. Fix one value `nums[i]`, then find pairs in the rest that sum to `-nums[i]` with two pointers from both ends. Skip repeated values so the same triplet is not reported twice.",
    "hint_zh": "先排序。固定一个值 `nums[i]`，再用从两端出发的双指针在其余部分中找和为 `-nums[i]` 的两个数。跳过重复的值，避免同一个三元组被输出两次。",
    "cases": [
        {"args": [[2, -2, 0, 1, -1]], "expected": [[-2, 0, 2], [-1, 0, 1]]},
        {"args": [[4, -4, 4, -4]], "expected": []},
        # The 0 appears twice, so [-1, 0, 1] is reachable two ways; it must be listed once
        {"args": [[-1, 0, 1, 0]], "expected": [[-1, 0, 1]]},
        # Four zeros give one triplet; a scan without duplicate skipping reports [0, 0, 0] twice
        {"args": [[0, 0, 0, 0]], "expected": [[0, 0, 0]]},
        {"args": [[-2, -2, -2, 4, 4, -4]], "expected": [[-2, -2, 4]]},
        {"args": [[-5, 3, 2]], "expected": [[-5, 2, 3]]},
        {"args": [[1, 2, 3]], "expected": []},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums = sorted(nums)
        res = []
        for i in range(len(nums) - 2):
            if nums[i] > 0:
                break  # sorted: no later triple can sum to zero
            if i > 0 and nums[i] == nums[i - 1]:
                continue  # this first value was already used
            lo, hi = i + 1, len(nums) - 1
            while lo < hi:
                total = nums[i] + nums[lo] + nums[hi]
                if total < 0:
                    lo += 1
                elif total > 0:
                    hi -= 1
                else:
                    res.append([nums[i], nums[lo], nums[hi]])
                    lo += 1
                    hi -= 1
                    while lo < hi and nums[lo] == nums[lo - 1]:
                        lo += 1
                    while lo < hi and nums[hi] == nums[hi + 1]:
                        hi -= 1
        return res''',
    "explanation": "Sort, then fix the first value `nums[i]` and run a two-sum search with two pointers over the rest. Sorting makes two things work: the pointers can move by comparing the sum with zero, and once `nums[i] > 0` no later triple can reach zero, so the loop stops. Skipping equal values at `i`, and at both pointers after a match, means each distinct triplet is reported exactly once.\n\nTime O(n²), space O(1) besides the sort and the output.",
    "explanation_zh": "先排序，然后固定第一个值 `nums[i]`，在其后的部分用双指针做两数之和查找。排序让指针可以根据和与 0 的大小关系移动；一旦 `nums[i] > 0`，之后的三元组都无法和为零，循环即可停止。找到一组答案后，跳过 `i` 处以及两个指针处相同的值，保证每个不同的三元组只输出一次。\n\n时间 O(n²)，除排序与输出外空间 O(1)。",
}
