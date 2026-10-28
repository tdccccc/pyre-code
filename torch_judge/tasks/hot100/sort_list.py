"""Sort List (LeetCode 148)."""

import random


def _large_case():
    # 50000 nodes, the constraint limit. Insertion or bubble sort on a linked list is quadratic
    # (about 10^9 steps here), while merge sort does about 8·10^5 steps
    rng = random.Random(148)
    values = [rng.randint(-10**5, 10**5) for _ in range(50000)]
    return {"args": [values], "expected": sorted(values)}


def _random_case(rng):
    return {"args": [[rng.randint(-9, 9) for _ in range(rng.randint(0, 8))]]}


TASK = {
    "title": "Sort List",
    "title_zh": "排序链表",
    "difficulty": "Medium",
    "leetcode_id": 148,
    "leetcode_slug": "sort-list",
    "category": "Linked List",
    "function_name": "sortList",
    "params": [("head", "ListNode")],
    "returns": "ListNode",
    "description_en": "Given the head of a linked list, return the list sorted in ascending order by value.\n\nThe sort must run in O(n log n) time.\n\n**Example 1:** `head = [9,-2,7,7,0]` → `[-2,0,7,7,9]`\n\n**Example 2:** `head = [1,1,0]` → `[0,1,1]`\n\n**Constraints:**\n- The list has `0` to `50000` nodes\n- $-10^{5} \\le \\texttt{Node.val} \\le 10^{5}$\n\n**Follow-up:** can you sort it using only O(1) extra memory (no recursion stack)?",
    "description_zh": "给定链表的头节点 `head`，返回按值升序排好的链表。\n\n排序的时间复杂度必须为 O(n log n)。\n\n**示例 1：** `head = [9,-2,7,7,0]` → `[-2,0,7,7,9]`\n\n**示例 2：** `head = [1,1,0]` → `[0,1,1]`\n\n**约束：**\n- 链表节点数在 `0` 到 `50000` 之间\n- $-10^{5} \\le \\texttt{Node.val} \\le 10^{5}$\n\n**进阶：** 你能只使用 O(1) 的额外内存（不用递归栈）完成排序吗？",
    "hint": "Merge sort suits linked lists: find the middle with slow and fast pointers, cut the list in two, sort each half, then merge the two sorted halves. Merging needs no extra array.",
    "hint_zh": "归并排序很适合链表：用快慢指针找中点，把链表从中间断开，分别排序两半，再合并两个有序链表。合并时不需要额外的数组。",
    "cases": [
        {"args": [[9, -2, 7, 7, 0]], "expected": [-2, 0, 7, 7, 9]},
        {"args": [[1, 1, 0]], "expected": [0, 1, 1]},
        {"args": [[]], "expected": []},
        {"args": [[4]], "expected": [4]},
        # Two nodes: the split has to cut the list into two single nodes
        {"args": [[2, 1]], "expected": [1, 2]},
        {"args": [[1, 2, 3, 4, 5, 6]], "expected": [1, 2, 3, 4, 5, 6]},
        {"args": [[6, 5, 4, 3, 2, 1]], "expected": [1, 2, 3, 4, 5, 6]},
        {"args": [[0, -100000, 100000, 0]], "expected": [-100000, 0, 0, 100000]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if head is None or head.next is None:
            return head
        # Slow ends at the last node of the left half; fast moves two steps for each step of slow
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        right, slow.next = slow.next, None  # cut the list in two
        return self.merge(self.sortList(head), self.sortList(right))

    def merge(self, a: Optional[ListNode], b: Optional[ListNode]) -> Optional[ListNode]:
        dummy = tail = ListNode()
        while a and b:
            if a.val <= b.val:
                tail.next, a = a, a.next
            else:
                tail.next, b = b, b.next
            tail = tail.next
        tail.next = a or b  # whichever list still has nodes
        return dummy.next''',
    "explanation": "Top-down merge sort. Find the middle of the list with slow and fast pointers, cut the list there, sort both halves recursively, and merge the two sorted lists. Merging relinks the existing nodes, so no new nodes or arrays are needed.\n\nTime O(n log n), space O(log n) for the recursion stack.",
    "explanation_zh": "自顶向下的归并排序。用快慢指针找到链表中点，在该处断开，递归排序两半，再合并两个有序链表。合并时只是重新连接已有的节点，不需要新建节点或数组。\n\n时间 O(n log n)，递归栈空间 O(log n)。",
}
