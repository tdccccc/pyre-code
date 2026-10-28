"""Linked List Cycle II (LeetCode 142)."""

import random


def _cycle_start(n: int, pos: int) -> int:
    """Walk node i -> i + 1 and the last node -> pos. Returns the index of the first node
    reached twice (the cycle's entry), or -1 when the walk ends (no cycle)."""
    seen, i = set(), 0
    while i >= 0 and i not in seen:
        seen.add(i)
        i = i + 1 if i < n - 1 else pos
    return i


def _large_case():
    # 10^4 nodes (the constraint) with the cycle entered at index 3000. Restarting a walk from
    # the head for every node is ~10^8 Python steps, which times out; the two-pointer method is linear.
    rng = random.Random(142)
    n = 10**4
    values = [rng.randint(-10**5, 10**5) for _ in range(n)]
    pos = 3000
    return {"args": [values, pos], "expected": _cycle_start(n, pos)}


def _random_case(rng):
    n = rng.randint(0, 7)
    values = [rng.randint(0, 3) for _ in range(n)]
    pos = rng.randint(0, n - 1) if n and rng.random() < 0.5 else -1
    return {"args": [values, pos]}


TASK = {
    "title": "Linked List Cycle II",
    "title_zh": "环形链表 II",
    "difficulty": "Medium",
    "leetcode_id": 142,
    "leetcode_slug": "linked-list-cycle-ii",
    "category": "Linked List",
    "function_name": "detectCycle",
    "params": [("head", "ListNode")],
    "returns": "ListNode",
    # Plain data in the cases: the values, and the index the tail links back to (-1 for none)
    "inputs": [("head", "ListNode"), ("pos", "int")],
    # The answer is a node. The judge reports it by its 0-based index from `head`, -1 for none
    "output": "int",
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
    "finish": '''def finish(result, args):
    if result is None:
        return -1
    # Walk the list from head; stop on a repeat so a node outside the list cannot loop forever
    seen, node, index = set(), args[0], 0
    while node is not None and id(node) not in seen:
        if node is result:
            return index
        seen.add(id(node))
        node = node.next
        index += 1
    return "a node that is not in the list"''',
    "description_en": "Given the head of a linked list, return the node where the cycle begins: the first node that is reached again when you walk from `head` following `next` pointers. Return `null` if there is no cycle. Do not change the list.\n\nThe tests describe a list the same way as for the cycle-detection problem: `head` gives the values in order, and `pos` is the index of the node the tail links back to (`-1` for none). The answer is shown as the node's 0-based index from `head`, or `-1` when there is no cycle, so in ACM mode print that index.\n\n**Example 1:** `head = [2,7,4,1,6], pos = 3` → `3` (the cycle starts at the node holding `1`)\n\n**Example 2:** `head = [3], pos = 0` → `0` (the only node points to itself)\n\n**Constraints:**\n- The list has `0` to $10^{4}$ nodes\n- $-10^{5} \\le \\texttt{Node.val} \\le 10^{5}$\n- `pos` is `-1` or a valid node index\n\n**Follow-up:** once the two pointers meet, how can you find where the cycle starts without a set of visited nodes?",
    "description_zh": "给定链表的头节点 `head`，返回环的入口节点：从 `head` 出发沿 `next` 指针前进，第一个被再次到达的节点。如果没有环，返回 `null`。不要修改链表。\n\n测试用例的表示方式与判断环的题目相同：`head` 给出按顺序排列的节点值，`pos` 是尾节点所连接的节点下标（无环为 `-1`）。答案以该入口节点从 `head` 起算的 0 开始下标显示（无环时为 `-1`），因此在 ACM 模式下请输出这个下标。\n\n**示例 1：** `head = [2,7,4,1,6], pos = 3` → `3`（环从值为 `1` 的节点开始）\n\n**示例 2：** `head = [3], pos = 0` → `0`（唯一的节点指向自己）\n\n**约束：**\n- 链表节点数在 `0` 到 $10^{4}$ 之间\n- $-10^{5} \\le \\texttt{Node.val} \\le 10^{5}$\n- `pos` 为 `-1` 或合法的节点下标\n\n**进阶：** 两个指针相遇之后，你能不用记录访问节点的集合，就找到环的起点吗？",
    "hint": "Once a fast and a slow pointer meet inside the cycle, restart one pointer at the head and step both one node at a time. They meet at the entry.",
    "hint_zh": "快慢指针在环内相遇后，把其中一个指针放回 `head`，两者都每次走一步，再次相遇的位置就是入口。",
    "cases": [
        {"args": [[2, 7, 4, 1, 6], 3], "expected": 3},
        {"args": [[3], 0], "expected": 0},
        {"args": [[5, 8], -1], "expected": -1},
        # Equal values: a value-based answer would point at index 0
        {"args": [[4, 4, 4, 4], 2], "expected": 2},
        # The cycle starts at the head
        {"args": [[1, 2, 3, 4, 5], 0], "expected": 0},
        # The head is outside the cycle
        {"args": [[3, 5, 8, 6, 2], 2], "expected": 2},
        # Where the fast and slow pointers meet is not the entry: returning the meeting node fails
        {"args": [[10, 20, 30, 40, 50, 60, 70], 3], "expected": 3},
        {"args": [[], -1], "expected": -1},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                # Restart one pointer at the head. Both now move one node at a time and
                # meet exactly at the start of the cycle
                entry = head
                while entry is not slow:
                    entry = entry.next
                    slow = slow.next
                return entry
        return None''',
    "explanation": "Floyd's algorithm in two phases. Phase 1 is the cycle check from the previous problem: if there is a cycle, the pointers meet at some node inside it. Phase 2 puts one pointer back at `head` and keeps the other at the meeting point, moving both one step at a time. Let `a` be the distance from `head` to the entry, `b` the distance from the entry to the meeting point, and `C` the cycle length. When the slow pointer met the fast one, the fast one had covered twice the slow one's distance, so `a + b` is a multiple of `C`. Walking `a` more steps from the meeting point therefore lands exactly on the entry.\n\nTime O(n), space O(1).",
    "explanation_zh": "Floyd 算法分两个阶段。第一阶段与上一题相同：若有环，两个指针会在环内某个节点相遇。第二阶段把一个指针放回 `head`，另一个留在相遇点，两者都每次走一步。设 `a` 为 `head` 到入口的距离，`b` 为入口到相遇点的距离，`C` 为环长。慢指针走到相遇点时，快指针走过的距离是慢指针的两倍，所以 `a + b` 是 `C` 的整数倍。从相遇点再走 `a` 步，正好绕到入口。\n\n时间 O(n)，空间 O(1)。",
}
