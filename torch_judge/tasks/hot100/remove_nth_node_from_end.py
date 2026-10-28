"""Remove Nth Node From End of List (LeetCode 19)."""

import random


def _large_case():
    # 30 nodes (the constraint). Nothing can time out at this size; the case checks the
    # removal of an interior node at full length.
    rng = random.Random(19)
    values = [rng.randint(0, 100) for _ in range(30)]
    n = rng.randint(1, 30)
    i = len(values) - n
    return {"args": [values, n], "expected": values[:i] + values[i + 1:]}


def _random_case(rng):
    values = [rng.randint(0, 9) for _ in range(rng.randint(1, 7))]
    return {"args": [values, rng.randint(1, len(values))]}


TASK = {
    "title": "Remove Nth Node From End of List",
    "title_zh": "删除链表的倒数第 N 个结点",
    "difficulty": "Medium",
    "leetcode_id": 19,
    "leetcode_slug": "remove-nth-node-from-end-of-list",
    "category": "Linked List",
    "function_name": "removeNthFromEnd",
    "params": [("head", "ListNode"), ("n", "int")],
    "returns": "ListNode",
    "description_en": "Given the head of a linked list and a positive integer `n`, remove the `n`-th node counted from the end of the list, and return the head of the updated list. Counting from the end, the last node is the 1st.\n\n**Example 1:** `head = [6,3,8,2,9], n = 2` → `[6,3,8,9]` (the 2nd node from the end holds `2`)\n\n**Example 2:** `head = [4,10], n = 2` → `[10]`\n\n**Constraints:**\n- The list has `1` to `30` nodes\n- $0 \\le \\texttt{Node.val} \\le 100$\n- $1 \\le \\texttt{n} \\le \\texttt{the}\\texttt{number}\\texttt{of}\\texttt{nodes}$\n\n**Follow-up:** can you do it without first measuring the length of the list?",
    "description_zh": "给定链表的头节点 `head` 和正整数 `n`，删除链表中从末尾数起的第 `n` 个节点，并返回更新后链表的头节点。从末尾数起，最后一个节点是第 1 个。\n\n**示例 1：** `head = [6,3,8,2,9], n = 2` → `[6,3,8,9]`（从末尾数第 2 个节点的值是 `2`）\n\n**示例 2：** `head = [4,10], n = 2` → `[10]`\n\n**约束：**\n- 链表节点数在 `1` 到 `30` 之间\n- $0 \\le \\texttt{Node.val} \\le 100$\n- `1 <= n <= 链表节点数`\n\n**进阶：** 你能不先遍历一遍求出链表长度，就完成删除吗？",
    "hint": "Put a dummy node before the head. Keep two pointers n + 1 nodes apart: when the fast one reaches the end, the slow one stops just before the node to delete.",
    "hint_zh": "在头节点前放一个哑节点。让两个指针保持 n + 1 个节点的间隔，快指针走到末尾时，慢指针正好停在待删节点之前。",
    "cases": [
        {"args": [[6, 3, 8, 2, 9], 2], "expected": [6, 3, 8, 9]},
        # Removing the head: a version without the dummy node must special-case this
        {"args": [[4, 10], 2], "expected": [10]},
        {"args": [[7, 1, 5], 1], "expected": [7, 1]},
        {"args": [[8, 2, 6, 5], 4], "expected": [2, 6, 5]},
        {"args": [[9], 1], "expected": []},
        # Equal values: the node to remove is the second 1, not the first
        {"args": [[3, 1, 4, 1, 5], 4], "expected": [3, 4, 1, 5]},
        {"args": [[1, 2, 3, 4], 3], "expected": [1, 3, 4]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)  # a dummy node in front, so removing the head needs no special case
        slow = fast = dummy
        # fast starts n + 1 nodes ahead of slow, so slow stops just before the node to remove
        for _ in range(n + 1):
            fast = fast.next
        while fast:
            slow = slow.next
            fast = fast.next
        slow.next = slow.next.next
        return dummy.next''',
    "explanation": "Use a dummy node in front of the head, so removing the first real node is no different from removing any other. Move `fast` `n + 1` steps from the dummy, then advance `fast` and `slow` together until `fast` runs off the end. The gap of `n + 1` leaves `slow` just before the target, which is then unlinked. One pass, no need to know the length beforehand.\n\nTime O(L), space O(1), where L is the list length.",
    "explanation_zh": "在头节点前放一个哑节点，这样删除第一个真实节点与删除其他节点没有区别。`fast` 从哑节点出发先走 `n + 1` 步，然后 `fast` 与 `slow` 同步前进，直到 `fast` 走出末尾。两者保持 `n + 1` 的间隔，使 `slow` 恰好停在待删节点之前，直接跳过它即可。只需遍历一次，无需事先知道长度。\n\n时间 O(L)，空间 O(1)，L 为链表长度。",
}
