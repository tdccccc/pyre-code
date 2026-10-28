"""LRU Cache (LeetCode 146)."""

import random
from collections import OrderedDict


def _large_case():
    # At the constraint limits. An O(capacity) list-based cache still finishes in about a
    # second here (list.remove runs in C), so this checks scale, not strict O(1).
    rng = random.Random(146)
    capacity = 3000
    ops, args, expected = ["LRUCache"], [[capacity]], [None]
    cache: OrderedDict = OrderedDict()
    for _ in range(200000):
        key = rng.randint(0, 4000)
        if rng.random() < 0.5:
            value = rng.randint(0, 10**5)
            ops.append("put")
            args.append([key, value])
            expected.append(None)
            cache[key] = value
            cache.move_to_end(key)
            if len(cache) > capacity:
                cache.popitem(last=False)
        else:
            ops.append("get")
            args.append([key])
            if key in cache:
                cache.move_to_end(key)
                expected.append(cache[key])
            else:
                expected.append(-1)
    return {"ops": ops, "args": args, "expected": expected}


def _random_case(rng):
    # Capacity 2 over four keys, so recency decides most evictions. Measured against
    # mutants: about 20% of these cases expose a get or put that skips the refresh.
    ops, args = ["LRUCache"], [[2]]
    for _ in range(rng.randint(15, 25)):
        key = rng.randint(0, 3)
        if rng.random() < 0.5:
            ops.append("put")
            args.append([key, rng.randint(0, 99)])
        else:
            ops.append("get")
            args.append([key])
    return {"ops": ops, "args": args}


TASK = {
    "title": "LRU Cache",
    "title_zh": "LRU 缓存",
    "difficulty": "Medium",
    "leetcode_id": 146,
    "leetcode_slug": "lru-cache",
    "category": "Linked List",
    "function_name": "LRUCache",
    "design": {
        "class": "LRUCache",
        "init": {"capacity": "int"},
        "methods": {
            "get": ({"key": "int"}, "int"),
            "put": ({"key": "int", "value": "int"}, "void"),
        },
    },
    "description_en": "Design a key-value cache with a fixed `capacity` that evicts the **least recently used** key when it is full. Implement the class `LRUCache`:\n\n- `LRUCache(capacity)` creates an empty cache holding at most `capacity` keys\n- `get(key)` returns the value for `key`, or `-1` if it is absent; a successful get counts as a use\n- `put(key, value)` inserts or updates `key`; it counts as a use. If this makes the cache exceed `capacity`, remove the least recently used key\n\nBoth `get` and `put` must run in **O(1)** average time.\n\n**Example:** capacity 2: `put(1,10)`, `put(2,20)`, `get(1)` → `10`, `put(3,30)` evicts key 2, `get(2)` → `-1`\n\n**Constraints:**\n- `1 <= capacity <= 3000`\n- `0 <= key <= 10^4`, `0 <= value <= 10^5`\n- At most `2 * 10^5` calls to `get` and `put`",
    "description_zh": "设计一个容量固定为 `capacity` 的键值缓存，满了以后淘汰**最久未使用**的键。实现 `LRUCache` 类：\n\n- `LRUCache(capacity)`：创建一个最多存放 `capacity` 个键的空缓存\n- `get(key)`：返回 `key` 对应的值，不存在则返回 `-1`；成功读取算作一次使用\n- `put(key, value)`：插入或更新 `key`，算作一次使用。如果因此超出 `capacity`，删除最久未使用的键\n\n`get` 和 `put` 的平均时间复杂度都必须是 **O(1)**。\n\n**示例：** 容量为 2：`put(1,10)`、`put(2,20)`、`get(1)` → `10`，`put(3,30)` 淘汰键 2，`get(2)` → `-1`\n\n**约束：**\n- `1 <= capacity <= 3000`\n- `0 <= key <= 10^4`，`0 <= value <= 10^5`\n- `get` 和 `put` 总共最多调用 `2 * 10^5` 次",
    "hint": "A dict gives O(1) lookup; a doubly linked list gives O(1) \"move to most recent\" and \"drop least recent\". Store list nodes in the dict. (collections.OrderedDict bundles both.)",
    "hint_zh": "字典提供 O(1) 查找，双向链表提供 O(1) 的「移到最近使用」和「删除最久未使用」。字典里存链表节点。（collections.OrderedDict 把两者合在了一起。）",
    "cases": [
        {
            "ops": ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"],
            "args": [[2], [1, 10], [2, 20], [1], [3, 30], [2], [4, 40], [1], [3], [4]],
            "expected": [None, None, None, 10, None, -1, None, -1, 30, 40],
        },
        {
            "ops": ["LRUCache", "put", "get", "put", "get", "get"],
            "args": [[1], [2, 1], [2], [3, 2], [2], [3]],
            "expected": [None, None, 1, None, -1, 2],
        },
        {
            # Updating an existing key refreshes it, so key 2 is evicted instead of key 1
            "ops": ["LRUCache", "put", "put", "put", "put", "get", "get", "get"],
            "args": [[2], [1, 1], [2, 2], [1, 5], [3, 3], [1], [2], [3]],
            "expected": [None, None, None, None, None, 5, -1, 3],
        },
        {
            "ops": ["LRUCache", "get", "put", "get", "put", "put", "get", "get"],
            "args": [[2], [7], [7, 70], [7], [8, 80], [9, 90], [7], [9]],
            "expected": [None, -1, None, 70, None, None, -1, 90],
        },
        {
            # A get refreshes recency: key 1 survives, key 2 is evicted
            "ops": ["LRUCache", "put", "put", "get", "put", "get", "get", "get"],
            "args": [[2], [1, 1], [2, 2], [1], [3, 3], [2], [3], [1]],
            "expected": [None, None, None, 1, None, -1, 3, 1],
        },
        {
            # Re-putting the only key at capacity 1 must update in place, not evict it
            "ops": ["LRUCache", "put", "put", "get", "put", "get", "get"],
            "args": [[1], [5, 1], [5, 2], [5], [6, 3], [5], [6]],
            "expected": [None, None, None, 2, None, -1, 3],
        },
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Node:
    def __init__(self, key=0, value=0):
        self.key, self.value = key, value
        self.prev = self.next = None


class LRUCache:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.nodes = {}
        # Sentinels: head.next is the least recent, tail.prev the most recent
        self.head, self.tail = Node(), Node()
        self.head.next, self.tail.prev = self.tail, self.head

    def _unlink(self, node):
        node.prev.next, node.next.prev = node.next, node.prev

    def _append(self, node):
        node.prev, node.next = self.tail.prev, self.tail
        self.tail.prev.next = node
        self.tail.prev = node

    def get(self, key: int) -> int:
        node = self.nodes.get(key)
        if node is None:
            return -1
        self._unlink(node)
        self._append(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        node = self.nodes.get(key)
        if node:
            node.value = value
            self._unlink(node)
        else:
            if len(self.nodes) == self.capacity:
                lru = self.head.next
                self._unlink(lru)
                del self.nodes[lru.key]
            node = self.nodes[key] = Node(key, value)
        self._append(node)''',
    "explanation": "Hash map + doubly linked list ordered by recency. The map finds a key's node in O(1); unlinking a node and appending it at the tail are O(1) pointer updates. Sentinel head/tail nodes remove the empty-list special cases. On overflow, evict `head.next`, the least recently used node, and delete its key from the map (which is why nodes store their key).\n\nTime O(1) per operation, space O(capacity).",
    "explanation_zh": "哈希表 + 按使用时间排序的双向链表。哈希表 O(1) 找到键对应的节点；把节点摘下、再接到链表尾部都是 O(1) 的指针操作。头尾哨兵节点省去了空链表的特判。超出容量时淘汰 `head.next`（最久未使用的节点），并从哈希表删除它的键，所以节点里要存 key。\n\n每次操作时间 O(1)，空间 O(capacity)。",
}
