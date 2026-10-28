"""Copy List with Random Pointer (LeetCode 138)."""

import random


def _large_case():
    # 1000 nodes, the constraint limit, with random pointers in about 70% of the nodes.
    # The copy has the same structure as the input, so the expected answer is the input itself
    rng = random.Random(138)
    n = 1000
    pairs = [[rng.randint(0, 1000), rng.randrange(n) if rng.random() < 0.7 else None] for _ in range(n)]
    return {"args": [pairs], "expected": [list(p) for p in pairs]}


def _random_case(rng):
    n = rng.randint(0, 6)
    pairs = [[rng.randint(0, 9), rng.randrange(n) if rng.random() < 0.6 else None] for _ in range(n)]
    return {"args": [pairs]}


TASK = {
    "title": "Copy List with Random Pointer",
    "title_zh": "随机链表的复制",
    "difficulty": "Medium",
    "leetcode_id": 138,
    "leetcode_slug": "copy-list-with-random-pointer",
    "category": "Linked List",
    "function_name": "copyRandomList",
    "params": [("head", "Node")],
    "returns": "Node",
    # The answer is the copy in the list-of-pairs format, so the finish hook's result is
    # checked as a plain value. finish also rejects a copy that reuses original nodes
    "output": "Node",
    "finish": '''def finish(result, args):
    def nodes(head):
        out = []
        while head is not None:
            out.append(head)
            if len(out) > 10**5:
                raise ValueError("linked list too long (cycle?)")
            head = head.next
        return out
    original = {id(n) for n in nodes(args[0])}
    for n in nodes(result):
        # A next pointer or a random pointer that lands on an original node is a reuse
        if id(n) in original or (n.random is not None and id(n.random) in original):
            return "the copy reuses nodes of the original list"
    return random_list_to_array(result)''',
    "description_en": "A linked list of `n` nodes is given. Each node has a `val`, a `next` pointer to the following node (null for the last node), and a `random` pointer that points to any node of the list, or to null.\n\nMake a deep copy: `n` brand-new nodes with the same values, whose `next` and `random` pointers lead to the matching new nodes. Return the head of the copy. A copy that reuses any original node, or has a `random` pointer aimed at an original node, is rejected.\n\nThe input is written as `[[val, random_index], ...]`, one pair per node in list order, where `random_index` is the position of the node its `random` points to (`null` for none). The answer uses the same format.\n\n**Example 1:** `head = [[6,2],[1,null],[4,0]]` → `[[6,2],[1,null],[4,0]]`\n\n**Example 2:** `head = [[3,0]]` → `[[3,0]]` (its only node's `random` points to itself)\n\n**Constraints:**\n- The list has `0` to `1000` nodes\n- `0 <= Node.val <= 1000`\n- Each `random` is null or points to a node of the list",
    "description_zh": "给定一个有 `n` 个节点的链表。每个节点有 `val`、指向后一个节点的 `next` 指针（最后一个节点为 null），以及指向链表中任意节点或 null 的 `random` 指针。\n\n请对它做深拷贝：创建 `n` 个全新的节点，值与原节点相同，它们的 `next` 和 `random` 指向对应的新节点。返回拷贝后链表的头节点。如果拷贝复用了任何原始节点，或者某个 `random` 指针指向原始节点，都会被判错。\n\n输入写成 `[[val, random_index], ...]`，按链表顺序每个节点一对，其中 `random_index` 是 `random` 所指节点的位置（没有则为 `null`）。输出使用同样的格式。\n\n**示例 1：** `head = [[6,2],[1,null],[4,0]]` → `[[6,2],[1,null],[4,0]]`\n\n**示例 2：** `head = [[3,0]]` → `[[3,0]]`（唯一节点的 `random` 指向它自己）\n\n**约束：**\n- 链表节点数在 `0` 到 `1000` 之间\n- `0 <= Node.val <= 1000`\n- 每个 `random` 为 null 或指向链表中的某个节点",
    "hint": "Make the copies first, keeping a dictionary from each original node to its copy. Then set every copy's `next` and `random` by looking up the targets in that dictionary.",
    "hint_zh": "先创建所有拷贝节点，并用字典记录「原节点 → 拷贝节点」。然后通过这个字典查找目标，设置每个拷贝节点的 `next` 和 `random`。",
    "cases": [
        {"args": [[[6, 2], [1, None], [4, 0]]], "expected": [[6, 2], [1, None], [4, 0]]},
        {"args": [[[3, 0]]], "expected": [[3, 0]]},
        {"args": [[]], "expected": []},
        # No random pointers at all: a dictionary lookup of None must not fail
        {"args": [[[1, None], [2, None], [3, None]]], "expected": [[1, None], [2, None], [3, None]]},
        # Random pointers that only go backwards
        {"args": [[[5, 4], [4, 3], [3, 2], [2, 1], [1, 0]]], "expected": [[5, 4], [4, 3], [3, 2], [2, 1], [1, 0]]},
        # Several nodes pointing at the same random target
        {"args": [[[9, 2], [8, 2], [7, 1], [6, 1]]], "expected": [[9, 2], [8, 2], [7, 1], [6, 1]]},
        # A copy that copies "random" as a plain reference gets this wrong: the pointer leads into the original list
        {"args": [[[0, 1], [1, 0]]], "expected": [[0, 1], [1, 0]]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def copyRandomList(self, head: Optional[Node]) -> Optional[Node]:
        copies = {None: None}  # original node -> its copy; None maps to None
        cur = head
        while cur:
            copies[cur] = Node(cur.val)
            cur = cur.next
        cur = head
        while cur:
            copy = copies[cur]
            copy.next = copies[cur.next]
            copy.random = copies[cur.random]
            cur = cur.next
        return copies[head]''',
    "explanation": "Map each original node to its copy in a dictionary. The first pass creates every copy with its value. The second pass sets `next` and `random` by mapping each original pointer to its copy. Null maps to null (the dictionary holds `None: None`), so missing pointers need no special case. Nodes are keyed by identity, since `Node` objects compare by identity.\n\nTime O(n), space O(n). The O(1)-space method interleaves each copy right after its original, sets the random pointers from that, and then splits the two lists apart.",
    "explanation_zh": "用字典把每个原节点映射到它的拷贝节点。第一轮遍历创建所有拷贝节点并设置值。第二轮遍历通过字典把原节点的指针映射为拷贝节点的指针，从而设置 `next` 和 `random`。空指针映射为空（字典中有 `None: None`），所以缺失的指针不需要特殊处理。字典以节点对象为键，因为 `Node` 对象按身份比较。\n\n时间 O(n)，空间 O(n)。O(1) 空间的做法是把每个拷贝节点穿插在原节点之后，据此设置 random 指针，最后把两条链表拆开。",
}
