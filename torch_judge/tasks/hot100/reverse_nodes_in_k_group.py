"""Reverse Nodes in k-Group (LeetCode 25)."""

import random


def _reverse_groups(values: list, k: int) -> list:
    out = []
    for start in range(0, len(values), k):
        block = values[start:start + k]
        out += block[::-1] if len(block) == k else block
    return out


def _large_case():
    # 5000 nodes, the constraint limit, with k = 37 so the last group is incomplete (5 nodes).
    # The pointer approach is linear, and no naive approach here runs slowly at this size
    rng = random.Random(25)
    values = [rng.randint(0, 1000) for _ in range(5000)]
    return {"args": [values, 37], "expected": _reverse_groups(values, 37)}


# The mistake (reversing an incomplete last group) only shows when at least two nodes are left
# over, which needs k >= 3. So k is 3 or 4, and the leftover is 2 or more in most cases. The
# rest keep a leftover of 0 or 1 to check that full groups still reverse
def _random_case(rng):
    k = rng.randint(3, 4)
    if rng.random() < 0.2:
        leftover = rng.randint(2, k - 1)
    else:
        leftover = rng.randint(0, k - 1)
    values = [rng.randint(0, 9) for _ in range(rng.randint(1, 2) * k + leftover)]
    return {"args": [values, k]}


TASK = {
    "title": "Reverse Nodes in k-Group",
    "title_zh": "K 个一组翻转链表",
    "difficulty": "Hard",
    "leetcode_id": 25,
    "leetcode_slug": "reverse-nodes-in-k-group",
    "category": "Linked List",
    "function_name": "reverseKGroup",
    "params": [("head", "ListNode"), ("k", "int")],
    "returns": "ListNode",
    "description_en": "Given the head of a linked list and a positive integer `k`, reverse the nodes in groups of `k`, counting from the head, and return the new head.\n\nA group is reversed only when it has exactly `k` nodes. If fewer than `k` nodes are left at the end, that last part keeps its order. Change the links between nodes; do not just swap values.\n\n**Example 1:** `head = [4,7,1,9,6], k = 2` → `[7,4,9,1,6]`\n\n**Example 2:** `head = [2,5,8,3,6,1,4], k = 3` → `[8,5,2,1,6,3,4]`\n\n**Constraints:**\n- The list has `1` to `5000` nodes\n- `1 <= k <= 5000`, and `k` is at most the number of nodes\n- `0 <= Node.val <= 1000`\n\n**Follow-up:** can you use only O(1) extra memory?",
    "description_zh": "给定链表的头节点 `head` 和正整数 `k`，从头开始每 `k` 个节点分为一组，对每组进行反转，返回新的头节点。\n\n只有节点数恰好为 `k` 的组才会反转。末尾不足 `k` 个节点的部分保持原来的顺序。请通过修改节点之间的指针来完成，不要只交换节点的值。\n\n**示例 1：** `head = [4,7,1,9,6], k = 2` → `[7,4,9,1,6]`\n\n**示例 2：** `head = [2,5,8,3,6,1,4], k = 3` → `[8,5,2,1,6,3,4]`\n\n**约束：**\n- 链表节点数在 `1` 到 `5000` 之间\n- `1 <= k <= 5000`，且 `k` 不超过节点数\n- `0 <= Node.val <= 1000`\n\n**进阶：** 你能只使用 O(1) 的额外内存吗？",
    "hint": "From the group's start, look `k` nodes ahead. If the lookahead runs off the end, stop. Otherwise reverse those `k` nodes, join the reversed group to the part before it, and continue from the group's old first node. A dummy node in front avoids special cases for the first group.",
    "hint_zh": "从当前组的起点向后数 `k` 个节点。如果数不够，就停止。否则反转这 `k` 个节点，把反转后的组接到前面的部分上，然后从这一组原来的第一个节点继续。在前面加一个哑节点，第一组就不需要特殊处理。",
    "cases": [
        {"args": [[4, 7, 1, 9, 6], 2], "expected": [7, 4, 9, 1, 6]},
        {"args": [[2, 5, 8, 3, 6, 1, 4], 3], "expected": [8, 5, 2, 1, 6, 3, 4]},
        {"args": [[1], 1], "expected": [1]},
        # k = 1 reverses single nodes, so nothing changes
        {"args": [[1, 2, 3, 4, 5], 1], "expected": [1, 2, 3, 4, 5]},
        # k equals the length: one group that ends exactly at the tail
        {"args": [[1, 2, 3, 4], 4], "expected": [4, 3, 2, 1]},
        # A single leftover node at the end must stay in place
        {"args": [[1, 2, 3, 4, 5, 6, 7], 3], "expected": [3, 2, 1, 6, 5, 4, 7]},
        # A leftover of two nodes: reversing the incomplete last group too gives [3, 2, 1, 5, 4]
        {"args": [[1, 2, 3, 4, 5], 3], "expected": [3, 2, 1, 4, 5]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode(0, head)
        group_prev = dummy  # the node just before the group being handled
        while True:
            # Find the k-th node of this group; stop if the group is incomplete
            kth = group_prev
            for _ in range(k):
                kth = kth.next
                if kth is None:
                    return dummy.next
            group_next = kth.next
            # Reverse the k nodes, starting from the node after the group
            prev, cur = group_next, group_prev.next
            while cur is not group_next:
                cur.next, prev, cur = prev, cur, cur.next
            first = group_prev.next  # the old first node is now the group's last
            group_prev.next = kth    # the reversed group now starts at kth
            group_prev = first''',
    "explanation": "Handle one group at a time. For the group after `group_prev`, look `k` nodes ahead: if the lookahead runs off the end, the group is incomplete and the rest stays as it is. Otherwise reverse those `k` nodes with the usual pointer loop, which stops on the node just past the group. Then `group_prev` links to the group's new head, and the group's old first node, now its last, becomes the next `group_prev`.\n\nTime O(n), space O(1). Each node is visited a constant number of times.",
    "explanation_zh": "每次处理一组。对于 `group_prev` 之后的这一组，先向后看 `k` 个节点：如果越过了末尾，说明这组不完整，剩下的部分保持原样。否则用常见的指针反转循环反转这 `k` 个节点，循环在组后第一个节点处停止。然后 `group_prev` 连到反转后这组的新头节点，而这组原来的第一个节点现在成了末尾，它就是下一轮的 `group_prev`。\n\n时间 O(n)，空间 O(1)。每个节点只被访问常数次。",
}
