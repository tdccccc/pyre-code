"""Merge Two Sorted Lists (LeetCode 21)."""

import random


def _sorted_values(rng, n: int) -> list:
    return sorted(rng.randint(-5, 5) for _ in range(n))


def _large_case():
    # 50 nodes per list (the constraint). Nothing can time out at this size, so the case
    # checks the splicing and the leftover tail at full length.
    rng = random.Random(21)
    a = sorted(rng.randint(-100, 100) for _ in range(50))
    b = sorted(rng.randint(-100, 100) for _ in range(50))
    return {"args": [a, b], "expected": sorted(a + b)}


def _random_case(rng):
    return {"args": [_sorted_values(rng, rng.randint(0, 5)), _sorted_values(rng, rng.randint(0, 5))]}


TASK = {
    "title": "Merge Two Sorted Lists",
    "title_zh": "合并两个有序链表",
    "difficulty": "Easy",
    "leetcode_id": 21,
    "leetcode_slug": "merge-two-sorted-lists",
    "category": "Linked List",
    "function_name": "mergeTwoLists",
    "params": [("list1", "ListNode"), ("list2", "ListNode")],
    "returns": "ListNode",
    "description_en": "Two linked lists `list1` and `list2` each hold their values in non-decreasing order. Merge them into one sorted list by splicing together the existing nodes, and return its head. Either list may be empty.\n\n**Example 1:** `list1 = [2,5,8], list2 = [1,5,9,10]` → `[1,2,5,5,8,9,10]`\n\n**Example 2:** `list1 = [], list2 = [0]` → `[0]`\n\n**Constraints:**\n- Each list has `0` to `50` nodes\n- $-100 \\le \\texttt{Node.val} \\le 100$\n- Both lists are sorted in non-decreasing order\n\n**Follow-up:** can you do it while allocating no new nodes except one dummy head?",
    "description_zh": "两个链表 `list1` 和 `list2` 的值都按非递减顺序排列。把它们合并成一个有序链表，合并时复用原有的节点，返回新链表的头节点。两个链表都可能为空。\n\n**示例 1：** `list1 = [2,5,8], list2 = [1,5,9,10]` → `[1,2,5,5,8,9,10]`\n\n**示例 2：** `list1 = [], list2 = [0]` → `[0]`\n\n**约束：**\n- 每个链表有 `0` 到 `50` 个节点\n- $-100 \\le \\texttt{Node.val} \\le 100$\n- 两个链表都按非递减顺序排列\n\n**进阶：** 你能除了一个哑节点（dummy head）之外不分配任何新节点吗？",
    "hint": "Like merging two sorted arrays: keep a dummy head, always splice the smaller front node onto the tail, and attach whatever remains at the end.",
    "hint_zh": "和合并两个有序数组一样：用哑节点作头，每次把较小的前端节点接到尾部，最后把剩下的整段接上。",
    "cases": [
        {"args": [[2, 5, 8], [1, 5, 9, 10]], "expected": [1, 2, 5, 5, 8, 9, 10]},
        {"args": [[], [0]], "expected": [0]},
        {"args": [[4], []], "expected": [4]},
        {"args": [[], []], "expected": []},
        # Equal values in both lists
        {"args": [[1, 1, 1], [1, 1]], "expected": [1, 1, 1, 1, 1]},
        {"args": [[-3, 0, 7], [-5, 2]], "expected": [-5, -3, 0, 2, 7]},
        # The second list runs out first, so the first list's remaining nodes must be attached
        {"args": [[1, 2, 3], [4, 5]], "expected": [1, 2, 3, 4, 5]},
        # The first list runs out first, so the second list's remaining nodes must be attached
        {"args": [[6, 7], [1, 2, 3]], "expected": [1, 2, 3, 6, 7]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = tail = ListNode()
        while list1 and list2:
            if list1.val <= list2.val:
                tail.next, list1 = list1, list1.next
            else:
                tail.next, list2 = list2, list2.next
            tail = tail.next
        # At most one list still has nodes, and they are already sorted
        tail.next = list1 or list2
        return dummy.next''',
    "explanation": "The merge step of merge sort. A dummy head holds the result. Each step splices the smaller of the two front nodes onto the tail. When one list runs out, the rest of the other list is already sorted, so it is attached as a whole.\n\nTime O(m + n), space O(1) beyond the output nodes.",
    "explanation_zh": "与归并排序中的合并步骤相同。用哑节点作为结果链表的头；每一步把两个链表前端较小的节点接到尾部。其中一个链表耗尽后，另一个链表剩下的部分已经有序，整段接上即可。\n\n时间 O(m + n)，空间 O(1)（不计输出的节点）。",
}
