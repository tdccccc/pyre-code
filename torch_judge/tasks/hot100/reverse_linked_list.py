"""Reverse Linked List (LeetCode 206)."""

import random


def _large_case():
    rng = random.Random(206)
    values = [rng.randint(-5000, 5000) for _ in range(5000)]
    return {"args": [values], "expected": values[::-1]}


def _random_case(rng):
    return {"args": [[rng.randint(-9, 9) for _ in range(rng.randint(0, 8))]]}


TASK = {
    "title": "Reverse Linked List",
    "title_zh": "反转链表",
    "difficulty": "Easy",
    "leetcode_id": 206,
    "leetcode_slug": "reverse-linked-list",
    "category": "Linked List",
    "function_name": "reverseList",
    "params": [("head", "ListNode")],
    "returns": "ListNode",
    "description_en": "Given the head `head` of a singly linked list, reverse the list and return the new head.\n\n**Example 1:** `head = [7,3,9,2]` → `[2,9,3,7]`\n\n**Example 2:** `head = [4,8]` → `[8,4]`\n\n**Constraints:**\n- The list has `0` to `5000` nodes\n- `-5000 <= Node.val <= 5000`\n\n**Follow-up:** can you write both an iterative and a recursive version?",
    "description_zh": "给定单链表的头节点 `head`，把链表反转，返回反转后的头节点。\n\n**示例 1：** `head = [7,3,9,2]` → `[2,9,3,7]`\n\n**示例 2：** `head = [4,8]` → `[8,4]`\n\n**约束：**\n- 链表节点数在 `0` 到 `5000` 之间\n- `-5000 <= Node.val <= 5000`\n\n**进阶：** 你能分别写出迭代和递归两种做法吗？",
    "hint": "Walk the list keeping `prev` and `cur`. Save `cur.next` before pointing `cur.next` back at `prev`, then step both forward.",
    "hint_zh": "遍历链表，维护 `prev` 和 `cur`。先保存 `cur.next`，再把 `cur.next` 指回 `prev`，然后两个指针一起前进。",
    "cases": [
        {"args": [[7, 3, 9, 2]], "expected": [2, 9, 3, 7]},
        {"args": [[4, 8]], "expected": [8, 4]},
        {"args": [[]], "expected": []},
        {"args": [[5]], "expected": [5]},
        {"args": [[1, 1, 2, 2]], "expected": [2, 2, 1, 1]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev, cur = None, head
        while cur:
            nxt = cur.next
            cur.next = prev
            prev, cur = cur, nxt
        return prev''',
    "explanation": "Iterative pointer reversal. Hold `prev` (the already-reversed part) and `cur`. Each step saves `cur.next`, points `cur` back at `prev`, and advances. When `cur` runs off the end, `prev` is the new head.\n\nTime O(n), space O(1). The recursive version is O(n) stack space; in your own ACM programs, deep recursion may need `sys.setrecursionlimit`.",
    "explanation_zh": "迭代反转指针。维护 `prev`（已反转的部分）和 `cur`。每一步先保存 `cur.next`，把 `cur` 指回 `prev`，再前进。`cur` 走到末尾时，`prev` 就是新的头节点。\n\n时间 O(n)，空间 O(1)。递归写法需要 O(n) 的栈空间；在 ACM 模式下自己写程序时，递归太深可能需要调大 `sys.setrecursionlimit`。",
}
