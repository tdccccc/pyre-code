"""Swap Nodes in Pairs (LeetCode 24)."""

import random


def _swap_pairs(values: list) -> list:
    out = list(values)
    for i in range(0, len(out) - 1, 2):
        out[i], out[i + 1] = out[i + 1], out[i]
    return out


def _large_case():
    # 100 nodes, the constraint limit. Any reasonable approach is linear here, so this checks
    # correctness at the largest allowed size rather than speed
    rng = random.Random(24)
    values = [rng.randint(0, 100) for _ in range(100)]
    return {"args": [values], "expected": _swap_pairs(values)}


def _random_case(rng):
    return {"args": [[rng.randint(0, 9) for _ in range(rng.randint(0, 7))]]}


TASK = {
    "title": "Swap Nodes in Pairs",
    "title_zh": "两两交换链表中的节点",
    "difficulty": "Medium",
    "leetcode_id": 24,
    "leetcode_slug": "swap-nodes-in-pairs",
    "category": "Linked List",
    "function_name": "swapPairs",
    "params": [("head", "ListNode")],
    "returns": "ListNode",
    "description_en": "Given the head of a singly linked list, swap every two adjacent nodes and return the head of the changed list. Swap the nodes themselves, not just their values.\n\nIf the list has an odd number of nodes, the final node has no partner and stays where it is.\n\n**Example 1:** `head = [5,6,7,8,9]` → `[6,5,8,7,9]`\n\n**Example 2:** `head = [4,9]` → `[9,4]`\n\n**Constraints:**\n- The list has `0` to `100` nodes\n- $0 \\le \\texttt{Node.val} \\le 100$",
    "description_zh": "给定单链表的头节点 `head`，两两交换相邻的节点，返回交换后链表的头节点。必须交换节点本身，而不能只交换节点里的值。\n\n如果节点数为奇数，最后一个节点没有配对的对象，保持原位即可。\n\n**示例 1：** `head = [5,6,7,8,9]` → `[6,5,8,7,9]`\n\n**示例 2：** `head = [4,9]` → `[9,4]`\n\n**约束：**\n- 链表节点数在 `0` 到 `100` 之间\n- $0 \\le \\texttt{Node.val} \\le 100$",
    "hint": "Handle one pair at a time with a `prev` pointer that stays just before the pair. Rewire `prev → b → a → rest`, then move `prev` to `a`. Stop when a pair is incomplete.",
    "hint_zh": "每次处理一对节点，用 `prev` 指针停在这一对的前面。把连接改成 `prev → b → a → 剩余部分`，然后让 `prev` 移到 `a`。遇到不完整的一对时停止。",
    "cases": [
        {"args": [[5, 6, 7, 8, 9]], "expected": [6, 5, 8, 7, 9]},
        {"args": [[4, 9]], "expected": [9, 4]},
        # Empty list: the pair loop must not touch head.next
        {"args": [[]], "expected": []},
        {"args": [[1]], "expected": [1]},
        # Even length: every node is in a pair
        {"args": [[1, 2, 3, 4, 5, 6]], "expected": [2, 1, 4, 3, 6, 5]},
        # Odd length: the last node has no partner. Code that swaps "head.next" blindly
        # crashes or drops it here
        {"args": [[9, 8, 7]], "expected": [8, 9, 7]},
        {"args": [[0, 0, 1, 2]], "expected": [0, 0, 2, 1]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def swapPairs(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode(0, head)  # a fixed node in front, so the first pair needs no special case
        prev = dummy
        while prev.next and prev.next.next:
            a, b = prev.next, prev.next.next
            a.next = b.next
            b.next = a
            prev.next = b
            prev = a
        return dummy.next''',
    "explanation": "Keep a `prev` pointer just before the pair being handled. It starts at a dummy node in front of the head, so the first pair is handled like any other. For a pair `a → b`, point `a` at whatever follows `b`, point `b` at `a`, and connect `prev` to `b`. Then move `prev` on to `a`, which now ends the swapped pair. A pair is handled only while both of its nodes exist, so a trailing odd node stays put.\n\nTime O(n), space O(1). A recursive version also works: swap the first two nodes, then recurse on the rest, using O(n) stack depth.",
    "explanation_zh": "用 `prev` 指针停在当前要处理的一对节点之前。`prev` 从头节点前面的哑节点开始，这样第一对和其他对的处理方式相同。对于一对 `a → b`，先让 `a` 指向 `b` 之后的部分，再让 `b` 指向 `a`，最后把 `prev` 接到 `b` 上。然后 `prev` 移到 `a`，此时 `a` 位于交换后这一对的末尾。只有两个节点都存在时才处理这一对，所以末尾单独的奇数节点保持不动。\n\n时间 O(n)，空间 O(1)。递归写法也可行：交换前两个节点，再对其余部分递归，栈深度为 O(n)。",
}
