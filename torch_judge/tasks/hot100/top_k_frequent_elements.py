"""Top K Frequent Elements (LeetCode 347)."""

import random
from collections import Counter


def _large_case():
    # 10^5 values. Sorting the counts also finishes well under 3 s, so this checks the answer at
    # the size limit rather than timing the O(n) bucket method against O(n log n). The boundary
    # between the k-th and (k+1)-th counts is made strict, so the top-k set is unique.
    rng = random.Random(347)
    nums = [rng.randint(-10**4, 10**4) for _ in range(10**5)]
    counts = Counter(nums)
    ranked = sorted(counts.values(), reverse=True)
    k = 20
    while k < len(ranked) and ranked[k - 1] == ranked[k]:
        k += 1
    top = sorted(counts, key=lambda v: -counts[v])[:k]
    return {"args": [nums, k], "expected": top}


def _random_case(rng):
    # The mistake is returning the first k distinct values in order of appearance. So the values
    # are shuffled with random counts over 3 to 5 distinct values, which often leaves the most
    # frequent value out of the first k. Retry until the k-th and (k+1)-th most frequent values
    # have different counts, so the top-k set is unique (a tie at the boundary would make either
    # tied value a valid answer)
    while True:
        values = rng.sample(range(-3, 4), rng.randint(3, 5))
        nums = [v for v in values for _ in range(rng.randint(1, 4))]
        rng.shuffle(nums)
        counts = sorted(Counter(nums).values(), reverse=True)
        k = rng.randint(1, len(counts))
        if k == len(counts) or counts[k - 1] > counts[k]:
            return {"args": [nums, k]}


TASK = {
    "title": "Top K Frequent Elements",
    "title_zh": "前 K 个高频元素",
    "difficulty": "Medium",
    "leetcode_id": 347,
    "leetcode_slug": "top-k-frequent-elements",
    "category": "Heap",
    "function_name": "topKFrequent",
    "params": [("nums", "int[]"), ("k", "int")],
    "returns": "int[]",
    "check": "unordered",
    "description_en": "Given an integer array `nums` and an integer `k`, return the `k` values that occur most often in `nums`. Return them in any order. The input is chosen so that the answer is unique: the `k`-th and `(k+1)`-th most frequent values never occur equally often.\n\n**Example 1:** `nums = [4,4,4,1,1,2,9], k = 2` → `[4,1]` (4 occurs three times, 1 occurs twice)\n\n**Example 2:** `nums = [5], k = 1` → `[5]`\n\n**Constraints:**\n- $1 \\le \\texttt{nums.length} \\le 10^{5}$\n- $-10^{4} \\le \\texttt{nums[i]} \\le 10^{4}$\n- `k` is between 1 and the number of distinct values in `nums`\n\n**Follow-up:** can you solve it in better than O(n log n) time?",
    "description_zh": "给定整数数组 `nums` 和整数 `k`，返回在 `nums` 中出现频率最高的 `k` 个值。可以按任意顺序返回。输入保证答案唯一：第 `k` 高频和第 $\\texttt{k} + 1$ 高频的值出现次数不会相同。\n\n**示例 1：** `nums = [4,4,4,1,1,2,9], k = 2` → `[4,1]`（4 出现了 3 次，1 出现了 2 次）\n\n**示例 2：** `nums = [5], k = 1` → `[5]`\n\n**约束：**\n- $1 \\le \\texttt{nums.length} \\le 10^{5}$\n- $-10^{4} \\le \\texttt{nums[i]} \\le 10^{4}$\n- `k` 在 1 到 `nums` 中不同值的个数之间\n\n**进阶：** 你能设计一个比 O(n log n) 更快的算法吗？",
    "hint": "Count with a dictionary. A value's count is at most `len(nums)`, so bucket the values by count, then read the buckets from the highest count down.",
    "hint_zh": "先用字典统计每个值的出现次数。次数最多为 `len(nums)`，所以可以按次数把值放进桶里，再从高次数的桶开始往下取。",
    "cases": [
        {"args": [[4, 4, 4, 1, 1, 2, 9], 2], "expected": [4, 1]},
        {"args": [[5], 1], "expected": [5]},
        # Values appear in the order 1, 2, 3, but 3 is the most frequent: returning the first k
        # distinct values in order of appearance gives [1, 2]
        {"args": [[1, 1, 2, 2, 2, 3, 3, 3, 3], 2], "expected": [3, 2]},
        {"args": [[-1, -1, 2, 2, 2, 3], 2], "expected": [2, -1]},
        # The count equals len(nums): a bucket indexed by count needs len(nums) + 1 slots
        {"args": [[3, 3, 3], 1], "expected": [3]},
        # k equals the number of distinct values
        {"args": [[1, 2, 3, 4], 4], "expected": [1, 2, 3, 4]},
        {"args": [[0, 0, 0, 0, 9, 8, 8, 7], 1], "expected": [0]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''from collections import Counter


class Solution:
    def topKFrequent(self, nums: List[int], k: int) -> List[int]:
        counts = Counter(nums)
        buckets = [[] for _ in range(len(nums) + 1)]  # bucket f holds the values seen f times
        for value, f in counts.items():
            buckets[f].append(value)
        result = []
        for f in range(len(nums), 0, -1):
            result.extend(buckets[f])
            if len(result) >= k:
                return result[:k]
        return result''',
    "explanation": "Count occurrences with a dictionary, then bucket the distinct values by count: `buckets[f]` holds every value that occurs `f` times. A count can be at most `len(nums)`, so there are `len(nums) + 1` buckets. Walk the buckets from the highest count down and collect values until there are `k`. Sorting the counts or keeping a min-heap of size `k` also works.\n\nTime O(n), space O(n).",
    "explanation_zh": "先用字典统计每个值的出现次数，再把不同的值按次数放进桶里：`buckets[f]` 存放所有出现了 `f` 次的值。次数最多为 `len(nums)`，所以需要 `len(nums) + 1` 个桶。从次数最高的桶开始往下收集，直到凑够 `k` 个。对次数排序，或者用大小为 `k` 的小根堆，也都能得到答案。\n\n时间 O(n)，空间 O(n)。",
}
