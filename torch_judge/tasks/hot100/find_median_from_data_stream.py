"""Find Median from Data Stream (LeetCode 295)."""

import bisect
import random


def _operations(rng, count: int, values: int) -> dict:
    # The first call is always addNum, so findMedian never sees an empty stream
    ops, args, size = ["MedianFinder"], [[]], 0
    for _ in range(count):
        if size == 0 or rng.random() < 0.5:
            ops.append("addNum")
            args.append([rng.randint(-values, values)])
            size += 1
        else:
            ops.append("findMedian")
            args.append([])
    return {"ops": ops, "args": args}


def _large_case():
    # 5 * 10^4 calls in total. The expected medians come from a sorted list kept with bisect,
    # which is independent of any heap solution.
    case = _operations(random.Random(295), 50000, 10**5)
    stream, expected = [], [None]
    for op, args in zip(case["ops"][1:], case["args"][1:]):
        if op == "addNum":
            bisect.insort(stream, args[0])
            expected.append(None)
        else:
            mid = len(stream) // 2
            if len(stream) % 2:
                expected.append(float(stream[mid]))
            else:
                expected.append((stream[mid - 1] + stream[mid]) / 2)
    case["expected"] = expected
    return case


def _random_case(rng):
    # Few distinct values, so duplicates and the boundary between halves come up often
    return _operations(rng, rng.randint(6, 14), 5)


TASK = {
    "title": "Find Median from Data Stream",
    "title_zh": "数据流的中位数",
    "difficulty": "Hard",
    "leetcode_id": 295,
    "leetcode_slug": "find-median-from-data-stream",
    "category": "Heap",
    "function_name": "MedianFinder",
    "design": {
        "class": "MedianFinder",
        "init": {},
        "methods": {
            "addNum": ({"num": "int"}, "void"),
            "findMedian": ({}, "float"),
        },
    },
    "description_en": "Design a structure that receives a stream of integers and reports the median of everything received so far. Implement the class `MedianFinder`:\n\n- `MedianFinder()` creates an empty stream\n- `addNum(num)` adds `num` to the stream\n- `findMedian()` returns the median of all values added so far. It is the middle value when the count is odd, or the average of the two middle values when it is even. Answers within 10^-5 of the exact value are accepted.\n\n**Example 1:** `addNum(4)`, `findMedian()` → `4`; `addNum(10)`, `findMedian()` → `7`; `addNum(-2)`, `findMedian()` → `4`\n\n**Example 2:** `addNum(-3)`, `addNum(-1)`, `addNum(-2)`, `findMedian()` → `-2`; `addNum(8)`, `findMedian()` → `-1.5`\n\n**Constraints:**\n- $-10^{5} \\le \\texttt{num} \\le 10^{5}$\n- `findMedian` is only called once at least one value has been added\n- At most $5 \\times 10^{4}$ calls in total to `addNum` and `findMedian`",
    "description_zh": "设计一个数据结构，持续接收整数流，并随时返回目前为止所有数的中位数。实现 `MedianFinder` 类：\n\n- `MedianFinder()`：创建一个空的数据流\n- `addNum(num)`：把 `num` 加入数据流\n- `findMedian()`：返回目前所有已加入的数的中位数。个数为奇数时是中间的那个数，为偶数时是中间两个数的平均值。与精确值相差不超过 10^-5 的答案均可接受。\n\n**示例 1：** `addNum(4)`、`findMedian()` → `4`；`addNum(10)`、`findMedian()` → `7`；`addNum(-2)`、`findMedian()` → `4`\n\n**示例 2：** `addNum(-3)`、`addNum(-1)`、`addNum(-2)`、`findMedian()` → `-2`；`addNum(8)`、`findMedian()` → `-1.5`\n\n**约束：**\n- $-10^{5} \\le \\texttt{num} \\le 10^{5}$\n- 只有在至少加入过一个数之后才会调用 `findMedian`\n- `addNum` 和 `findMedian` 的总调用次数不超过 $5 \\times 10^{4}$",
    "hint": "Keep two heaps: a max-heap for the smaller half and a min-heap for the larger half. Keep their sizes equal or differ by one, and the median is at the tops.",
    "hint_zh": "用两个堆：大根堆保存较小的一半，小根堆保存较大的一半。让两者大小相等或只差一，中位数就在两个堆的堆顶。",
    "cases": [
        {
            "ops": ["MedianFinder", "addNum", "findMedian", "addNum", "findMedian", "addNum", "findMedian"],
            "args": [[], [4], [], [10], [], [-2], []],
            "expected": [None, None, 4.0, None, 7.0, None, 4.0],
        },
        {
            "ops": ["MedianFinder", "addNum", "addNum", "addNum", "findMedian", "addNum", "findMedian"],
            "args": [[], [-3], [-1], [-2], [], [8], []],
            "expected": [None, None, None, None, -2.0, None, -1.5],
        },
        {
            "ops": ["MedianFinder", "addNum", "findMedian"],
            "args": [[], [5], []],
            "expected": [None, None, 5.0],
        },
        # Descending input: a stream that only appends to one heap is not ordered
        {
            "ops": ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"],
            "args": [[], [9], [8], [], [1], []],
            "expected": [None, None, None, 8.5, None, 8.0],
        },
        # Duplicates: equal values on both sides of the middle
        {
            "ops": ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "addNum", "findMedian"],
            "args": [[], [6], [6], [], [6], [1], []],
            "expected": [None, None, None, 6.0, None, None, 6.0],
        },
        # Extreme values at both ends of the allowed range, so the middle is their average
        {
            "ops": ["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"],
            "args": [[], [-100000], [100000], [], [100000], []],
            "expected": [None, None, None, 0.0, None, 100000.0],
        },
        # Four values, then one more: the heaps must rebalance so the odd-count median is the
        # top of the larger heap
        {
            "ops": ["MedianFinder", "addNum", "addNum", "addNum", "addNum", "findMedian", "addNum", "findMedian"],
            "args": [[], [1], [2], [3], [4], [], [100], []],
            "expected": [None, None, None, None, None, 2.5, None, 3.0],
        },
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''import heapq


class MedianFinder:

    def __init__(self):
        self.low = []   # max-heap (values stored negated): the smaller half
        self.high = []  # min-heap: the larger half

    def addNum(self, num: int) -> None:
        heapq.heappush(self.low, -num)
        # Move the largest of the lower half up, so every low value is <= every high value
        heapq.heappush(self.high, -heapq.heappop(self.low))
        if len(self.high) > len(self.low):
            heapq.heappush(self.low, -heapq.heappop(self.high))

    def findMedian(self) -> float:
        if len(self.low) > len(self.high):
            return float(-self.low[0])
        return (-self.low[0] + self.high[0]) / 2''',
    "explanation": "Split the stream into two heaps: `low` is a max-heap of the smaller half and `high` a min-heap of the larger half. `addNum` pushes the value into `low`, moves the largest of `low` into `high`, and then, if `high` is larger, moves its smallest back. The sizes therefore stay equal or differ by one, and every value in `low` is at most every value in `high`. When the count is odd, the median is the top of `low`; when it is even, it is the average of the two tops.\n\nTime O(log n) per `addNum`, O(1) per `findMedian`, space O(n).",
    "explanation_zh": "把数据流分成两个堆：`low` 是保存较小一半的大根堆，`high` 是保存较大一半的小根堆。`addNum` 先把值压入 `low`，再把 `low` 中最大的值移到 `high`；如果此时 `high` 比 `low` 大，就把 `high` 中最小的值移回 `low`。这样两个堆的大小始终相等或只差一，并且 `low` 中的每个值都不大于 `high` 中的每个值。个数为奇数时，中位数是 `low` 的堆顶；为偶数时，是两个堆顶的平均值。\n\n每次 `addNum` 时间 O(log n)，每次 `findMedian` 时间 O(1)，空间 O(n)。",
}
