"""Merge k Sorted Lists (LeetCode 23)."""

import random


def _large_case():
    # 10^4 lists of one node each, the largest allowed total (10^4 nodes). Scanning every list
    # head at each step does about 10^8 comparisons, which takes about 2 s here, still under the
    # 3 s limit, and merging one list at a time is faster still. Within the constraints no naive
    # approach times out, so this case mainly checks scale. A heap does about 10^4 · log(10^4) work
    rng = random.Random(23)
    lists = [[rng.randint(-10**4, 10**4)] for _ in range(10**4)]
    expected = sorted(v for lst in lists for v in lst)
    return {"args": [lists], "expected": expected}


def _random_case(rng):
    # Small values, so equal values across lists come up often (the tie-breaking bug)
    lists = [sorted(rng.randint(1, 9) for _ in range(rng.randint(0, 3))) for _ in range(rng.randint(0, 3))]
    return {"args": [lists]}


TASK = {
    "title": "Merge k Sorted Lists",
    "title_zh": "合并 K 个升序链表",
    "difficulty": "Hard",
    "leetcode_id": 23,
    "leetcode_slug": "merge-k-sorted-lists",
    "category": "Linked List",
    "function_name": "mergeKLists",
    "params": [("lists", "ListNode[]")],
    "returns": "ListNode",
    "description_en": "You are given `lists`, an array of `k` linked lists. Each list is sorted in ascending order.\n\nMerge all the lists into one sorted linked list and return its head.\n\n**Example 1:** `lists = [[8,10],[3],[1,9,12]]` → `[1,3,8,9,10,12]`\n\n**Example 2:** `lists = [[],[-4,0]]` → `[-4,0]`\n\n**Constraints:**\n- `0 <= k <= 10^4`\n- Each list has at most `500` nodes, and the total number of nodes over all lists is at most `10^4`\n- `-10^4 <= Node.val <= 10^4`\n- Each list is sorted in ascending order",
    "description_zh": "给定数组 `lists`，其中包含 `k` 个链表，每个链表都已按升序排列。\n\n把所有链表合并为一个升序链表，返回合并后的头节点。\n\n**示例 1：** `lists = [[8,10],[3],[1,9,12]]` → `[1,3,8,9,10,12]`\n\n**示例 2：** `lists = [[],[-4,0]]` → `[-4,0]`\n\n**约束：**\n- `0 <= k <= 10^4`\n- 每个链表最多有 `500` 个节点，所有链表的节点总数不超过 `10^4`\n- `-10^4 <= Node.val <= 10^4`\n- 每个链表都按升序排列",
    "hint": "Keep one candidate per list in a min-heap, the current head of each list. Pop the smallest, append it to the result, and push the next node from the same list.",
    "hint_zh": "用一个小根堆保存每个链表当前的头节点（每个链表一个候选）。每次弹出最小的节点接到结果后面，再把它所在链表的下一个节点放入堆中。",
    "cases": [
        {"args": [[[8, 10], [3], [1, 9, 12]]], "expected": [1, 3, 8, 9, 10, 12]},
        {"args": [[[], [-4, 0]]], "expected": [-4, 0]},
        {"args": [[]], "expected": []},
        {"args": [[[]]], "expected": []},
        # Equal values in different lists: a heap of (value, node) tuples would compare nodes
        {"args": [[[1], [1], [1]]], "expected": [1, 1, 1]},
        {"args": [[[-5, -5, 2], [-6, 0], [7]]], "expected": [-6, -5, -5, 0, 2, 7]},
        {"args": [[[4, 4], [2, 4], [4]]], "expected": [2, 4, 4, 4, 4]},
        {"args": [[[10], [], [9], [], [0, 11]]], "expected": [0, 9, 10, 11]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''import heapq


class Solution:
    def mergeKLists(self, lists: List[Optional[ListNode]]) -> Optional[ListNode]:
        # Entries are (value, list index, node). The index breaks ties, so nodes are never compared
        heap = [(node.val, i, node) for i, node in enumerate(lists) if node]
        heapq.heapify(heap)
        dummy = tail = ListNode()
        while heap:
            _, i, node = heapq.heappop(heap)
            tail.next = node
            tail = node
            if node.next:
                heapq.heappush(heap, (node.next.val, i, node.next))
        return dummy.next''',
    "explanation": "Min-heap of the current head of every list. Each step pops the smallest head, appends that node to the result, and pushes the node after it, if any. Each list contributes at most one entry to the heap at a time, so the list index in the tuple already makes every entry unique, and the nodes themselves are never compared.\n\nTime O(N log k), where N is the total number of nodes and k the number of lists. Space O(k) for the heap. Merging the lists one pair at a time costs O(N k) in the worst case.",
    "explanation_zh": "用小根堆保存每个链表当前的头节点。每一步弹出最小的头节点，把它接到结果后面，再把它的下一个节点放入堆中（如果存在）。每个链表同一时刻最多只有一个节点在堆中，所以元组中的链表下标已经让每个条目唯一，节点本身永远不会被比较。\n\n时间 O(N log k)，其中 N 是所有节点总数，k 是链表个数；空间 O(k)。如果两两依次合并，最坏情况下时间为 O(N k)。",
}
