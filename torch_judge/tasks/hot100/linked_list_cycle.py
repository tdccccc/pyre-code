"""Linked List Cycle (LeetCode 141)."""

import random


def _cycle_start(n: int, pos: int) -> int:
    """Walk node i -> i + 1 and the last node -> pos. Returns the index of the first node
    reached twice, or -1 when the walk ends (no cycle)."""
    seen, i = set(), 0
    while i >= 0 and i not in seen:
        seen.add(i)
        i = i + 1 if i < n - 1 else pos
    return i


def _large_case():
    # 10^4 nodes (the constraint) with a cycle back to index 4000. Restarting a walk from the
    # head for every node is ~10^8 Python steps, which times out; the two-pointer pass is linear.
    rng = random.Random(141)
    n = 10**4
    values = [rng.randint(-10**5, 10**5) for _ in range(n)]
    pos = 4000
    return {"args": [values, pos], "expected": _cycle_start(n, pos) >= 0}


def _random_case(rng):
    # Small values repeat often: a check that compares values instead of nodes fails on
    # lists without a cycle
    n = rng.randint(0, 7)
    values = [rng.randint(0, 3) for _ in range(n)]
    pos = rng.randint(0, n - 1) if n and rng.random() < 0.5 else -1
    return {"args": [values, pos]}


TASK = {
    "title": "Linked List Cycle",
    "title_zh": "环形链表",
    "difficulty": "Easy",
    "leetcode_id": 141,
    "leetcode_slug": "linked-list-cycle",
    "category": "Linked List",
    "function_name": "hasCycle",
    "params": [("head", "ListNode")],
    "returns": "bool",
    # The judge builds the list from its values and links the tail back to node `pos`
    "inputs": [("head", "ListNode"), ("pos", "int")],
    "prepare": '''def prepare(args):
    head, pos = args
    if pos < 0:
        return [head]
    target = tail = head
    for _ in range(pos):
        target = target.next
    while tail.next:
        tail = tail.next
    tail.next = target
    return [head]''',
    "description_en": "Given the head of a linked list, return `true` if the list contains a cycle, meaning that following `next` pointers from some node leads back to a node already visited, and `false` otherwise. Compare nodes, not values: equal values in different nodes do not make a cycle.\n\nIn the test cases, `head` gives the values in list order, and `pos` is the index of the node the tail links back to (`-1` for no cycle).\n\n**Example 1:** `head = [5,9,1,7], pos = 2` → `true` (the last node links back to the node holding `1`)\n\n**Example 2:** `head = [8,4,6], pos = -1` → `false`\n\n**Constraints:**\n- The list has `0` to `10^4` nodes\n- `-10^5 <= Node.val <= 10^5`\n- `pos` is `-1` or a valid node index\n\n**Follow-up:** can you decide it with only two pointers, without remembering the nodes you have visited?",
    "description_zh": "给定链表的头节点 `head`，如果链表中存在环，返回 `true`，否则返回 `false`。所谓环，是指沿着 `next` 指针从某个节点出发，能够回到之前已经访问过的节点。比较的是节点本身而不是值：两个不同节点的值相等，并不构成环。\n\n测试用例中，`head` 给出按链表顺序排列的节点值，`pos` 是尾节点所连接的那个节点的下标（无环时为 `-1`）。\n\n**示例 1：** `head = [5,9,1,7], pos = 2` → `true`（最后一个节点指回值为 `1` 的节点）\n\n**示例 2：** `head = [8,4,6], pos = -1` → `false`\n\n**约束：**\n- 链表节点数在 `0` 到 `10^4` 之间\n- `-10^5 <= Node.val <= 10^5`\n- `pos` 为 `-1` 或合法的节点下标\n\n**进阶：** 你能只用两个指针判断，而不记录已经访问过的节点吗？",
    "hint": "Move a slow pointer one step and a fast pointer two steps. With a cycle they must meet inside it; without one, the fast pointer reaches the end first.",
    "hint_zh": "让慢指针每次走一步，快指针每次走两步。有环时二者必在环内相遇；无环时快指针会先走到末尾。",
    "cases": [
        {"args": [[5, 9, 1, 7], 2], "expected": True},
        {"args": [[8, 4, 6], -1], "expected": False},
        {"args": [[1], -1], "expected": False},
        # A single node that points at itself
        {"args": [[1], 0], "expected": True},
        # Equal values without a cycle: a value-based visited set reports a false cycle
        {"args": [[3, 3, 8, 5], -1], "expected": False},
        {"args": [[2, 2, 2], -1], "expected": False},
        # The cycle does not include the head
        {"args": [[1, 2, 3, 4], 1], "expected": True},
        # Cycle of length one at the tail
        {"args": [[1, 2, 3, 4, 5, 6], 5], "expected": True},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            # If there is a cycle, the fast pointer gains one node on the slow one each round
            if slow is fast:
                return True
        return False''',
    "explanation": "Floyd's cycle detection (tortoise and hare). The slow pointer moves one step and the fast pointer two. Without a cycle, the fast pointer reaches the end first. With a cycle, the gap between them shrinks by one each round, so they must meet inside the cycle.\n\nTime O(n), space O(1).",
    "explanation_zh": "Floyd 判圈算法（龟兔赛跑）。慢指针每次走一步，快指针每次走两步。无环时快指针先走到末尾；有环时两者的距离每轮缩小一，最终必在环内相遇。\n\n时间 O(n)，空间 O(1)。",
}
