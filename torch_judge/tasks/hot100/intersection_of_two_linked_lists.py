"""Intersection of Two Linked Lists (LeetCode 160)."""

import random


def _lists(only_a: list, only_b: list, shared: list) -> list:
    """Case args: both lists' values plus where each one joins the shared tail."""
    return [only_a + shared, only_b + shared, len(only_a), len(only_b)]


def _large_case():
    # 3 * 10^4 nodes per list joined near the end: comparing every pair of nodes is ~10^9 steps
    rng = random.Random(160)
    values = rng.sample(range(1, 10**5), 60010)
    only_a, only_b, shared = values[:29995], values[29995:59990], values[59990:]
    return {"args": _lists(only_a, only_b, shared), "expected": shared[0]}


def _random_case(rng):
    # Distinct values, so a node can only match by identity, never by an equal value
    values = rng.sample(range(1, 100), 12)
    a, b, s = rng.randint(0, 4), rng.randint(0, 4), rng.randint(0, 4)
    return {"args": _lists(values[:a], values[a:a + b], values[a + b:a + b + s])}


TASK = {
    "title": "Intersection of Two Linked Lists",
    "title_zh": "相交链表",
    "difficulty": "Easy",
    "leetcode_id": 160,
    "leetcode_slug": "intersection-of-two-linked-lists",
    "category": "Linked List",
    "function_name": "getIntersectionNode",
    "params": [("headA", "ListNode"), ("headB", "ListNode")],
    "returns": "ListNode",
    # The judge builds the two lists from plain data so that they really share their tail
    # nodes, then reports the returned node by its value (null when there is none)
    "inputs": [("listA", "int[]"), ("listB", "int[]"), ("skipA", "int"), ("skipB", "int")],
    "output": "int?",
    "prepare": '''def prepare(args):
    values_a, values_b, skip_a, skip_b = args
    head_a = build_list(values_a)
    shared = head_a
    for _ in range(skip_a):
        shared = shared.next
    head_b = build_list(values_b[:skip_b])
    if head_b is None:
        return [head_a, shared]
    tail = head_b
    while tail.next:
        tail = tail.next
    tail.next = shared
    return [head_a, head_b]''',
    "finish": '''def finish(result, args):
    if result is None:
        return None
    # It must be a node of both lists, not a copy that merely has the same value
    for head in args:
        node = head
        while node is not None and node is not result:
            node = node.next
        if node is None:
            return "a node that is not in both lists"
    return result.val''',
    "description_en": "Two singly linked lists `headA` and `headB` may merge: from some node on, they continue through the very same nodes to the end. Return the first node the two lists share, or `null` if they share none.\n\nThe lists have no cycles, and they must keep their original structure after the function returns.\n\n**How the tests describe a case:** `listA` and `listB` are the values of the two lists; the first `skipA` nodes of A and the first `skipB` nodes of B belong to one list only, and everything after them is shared. The expected answer is the value of the first shared node. Sharing is by node, not by value: two separate nodes holding equal values are not an intersection.\n\n**Example 1:** `listA = [3,9,6,2], listB = [5,6,2], skipA = 2, skipB = 1` → `6` (both lists end with the same nodes `6 → 2`)\n\n**Example 2:** `listA = [1,4], listB = [7], skipA = 2, skipB = 1` → `null`\n\n**Constraints:**\n- $1 \\le \\texttt{listA.length},\\ \\texttt{listB.length} \\le 3 \\times 10^{4}$\n- $1 \\le \\texttt{Node.val} \\le 10^{5}$\n- $0 \\le \\texttt{skipA} \\le \\texttt{listA.length}$, $0 \\le \\texttt{skipB} \\le \\texttt{listB.length}$\n- `skipA == listA.length` exactly when `skipB == listB.length` (no shared part)\n\n**Follow-up:** can you solve it in O(m + n) time with O(1) extra space?",
    "description_zh": "两个单链表 `headA` 和 `headB` 可能会相交：从某个节点开始，它们经过完全相同的节点一直走到末尾。返回两个链表共有的第一个节点；如果没有，返回 `null`。\n\n链表中没有环，函数返回后两个链表必须保持原来的结构。\n\n**用例的表示方式：** `listA` 和 `listB` 是两个链表的值；A 的前 `skipA` 个节点和 B 的前 `skipB` 个节点只属于各自的链表，之后的部分是共享的。期望答案是第一个共享节点的值。共享指的是同一个节点，而不是值相等：两个值相同的不同节点不算相交。\n\n**示例 1：** `listA = [3,9,6,2], listB = [5,6,2], skipA = 2, skipB = 1` → `6`（两个链表都以同样的节点 `6 → 2` 结尾）\n\n**示例 2：** `listA = [1,4], listB = [7], skipA = 2, skipB = 1` → `null`\n\n**约束：**\n- $1 \\le \\texttt{listA.length},\\ \\texttt{listB.length} \\le 3 \\times 10^{4}$\n- $1 \\le \\texttt{Node.val} \\le 10^{5}$\n- $0 \\le \\texttt{skipA} \\le \\texttt{listA.length}$，$0 \\le \\texttt{skipB} \\le \\texttt{listB.length}$\n- `skipA == listA.length` 当且仅当 `skipB == listB.length`（此时没有共享部分）\n\n**进阶：** 你能用 O(m + n) 时间、O(1) 额外空间解决吗？",
    "hint": "Let pointer a walk A then B, and pointer b walk B then A. Both cover the same total length, so they arrive at the first shared node together (or at the end, None, together).",
    "hint_zh": "让指针 a 先走 A 再走 B，指针 b 先走 B 再走 A。两者走过的总长度相同，所以会同时到达第一个共享节点（没有交点时同时走到末尾 None）。",
    "cases": [
        {"args": [[3, 9, 6, 2], [5, 6, 2], 2, 1], "expected": 6},
        {"args": [[1, 4], [7], 2, 1], "expected": None},
        # Equal values before the real junction: comparing values instead of nodes stops early
        {"args": [[4, 1, 8, 4, 5], [5, 6, 1, 8, 4, 5], 2, 3], "expected": 8},
        {"args": [[2, 6, 4], [1, 5], 3, 2], "expected": None},
        # One list is entirely the other's tail
        {"args": [[8, 3, 5], [3, 5], 1, 0], "expected": 3},
        {"args": [[7], [7], 0, 0], "expected": 7},
        {"args": [[1, 2, 3, 4, 5, 6, 7], [9, 7], 6, 1], "expected": 7},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def getIntersectionNode(self, headA: ListNode, headB: ListNode) -> Optional[ListNode]:
        a, b = headA, headB
        # Each pointer walks both lists once; they meet at the junction or at None
        while a is not b:
            a = a.next if a else headB
            b = b.next if b else headA
        return a''',
    "explanation": "Two pointers that switch lists. Say A has `x` nodes before the junction, B has `y`, and the shared tail has `z`. Pointer `a` walks A and then B; pointer `b` walks B and then A. After `x + z + y` steps both have reached the junction, because each has covered its own private part, the shared part and the other private part. With no junction (`z = 0`) both reach `None` after `x + y` steps, so the loop ends either way.\n\nTime O(m + n), space O(1). A hash set of A's nodes also works in O(m) space.",
    "explanation_zh": "交换链表的双指针。设 A 在交点前有 `x` 个节点，B 有 `y` 个，共享部分有 `z` 个。指针 `a` 先走 A 再走 B，指针 `b` 先走 B 再走 A。走了 `x + z + y` 步后两者都到达交点，因为各自都走完了自己的独有部分、共享部分和对方的独有部分。没有交点（`z = 0`）时，两者在 `x + y` 步后同时到达 `None`，所以循环一定会结束。\n\n时间 O(m + n)，空间 O(1)。也可以用哈希集合存下 A 的所有节点，空间 O(m)。",
}
