"""Min Stack (LeetCode 155)."""

import random


def _simulate(ops: list, args: list) -> list:
    stack, out = [], [None]
    for op, a in zip(ops[1:], args[1:]):
        if op == "push":
            stack.append(a[0])
            out.append(None)
        elif op == "pop":
            stack.pop()
            out.append(None)
        elif op == "top":
            out.append(stack[-1])
        else:
            out.append(min(stack))
    return out


def _operations(rng, count: int, values: int) -> dict:
    # Calls on an empty stack are never made, as the constraints promise
    ops, args, size = ["MinStack"], [[]], 0
    for _ in range(count):
        choice = rng.random() if size else 0.0
        if choice < 0.4:
            ops.append("push")
            args.append([rng.randint(-values, values)])
            size += 1
        elif choice < 0.6:
            ops.append("pop")
            args.append([])
            size -= 1
        else:
            ops.append(rng.choice(["top", "getMin"]))
            args.append([])
    return {"ops": ops, "args": args}


def _large_case():
    # 3 * 10^4 calls on a stack thousands deep. getMin as min(stack) still finishes (min runs
    # in C), so like LRU Cache this checks scale rather than strict O(1).
    case = _operations(random.Random(155), 22000, 2**31 - 1)
    case["ops"][1:1] = ["push"] * 8000
    case["args"][1:1] = [[v] for v in range(8000, 0, -1)]
    case["expected"] = _simulate(case["ops"], case["args"])
    return case


def _random_case(rng):
    # Few distinct values, so duplicate minimums (and popping one of them) come up often
    return _operations(rng, rng.randint(8, 16), 3)


TASK = {
    "title": "Min Stack",
    "title_zh": "最小栈",
    "difficulty": "Medium",
    "leetcode_id": 155,
    "leetcode_slug": "min-stack",
    "category": "Stack",
    "function_name": "MinStack",
    "design": {
        "class": "MinStack",
        "init": {},
        "methods": {
            "push": ({"val": "int"}, "void"),
            "pop": ({}, "void"),
            "top": ({}, "int"),
            "getMin": ({}, "int"),
        },
    },
    "description_en": "Design a stack that can also report its smallest element in constant time. Implement the class `MinStack`:\n\n- `MinStack()` creates an empty stack\n- `push(val)` puts `val` on top\n- `pop()` removes the top element\n- `top()` returns the top element\n- `getMin()` returns the smallest element currently in the stack\n\nEvery method must run in **O(1)** time.\n\n**Example:** `push(4)`, `push(1)`, `push(6)`, `getMin()` → `1`, `pop()`, `top()` → `1`, `pop()`, `getMin()` → `4`\n\n**Constraints:**\n- `-2^31 <= val <= 2^31 - 1`\n- `pop`, `top` and `getMin` are only called on a non-empty stack\n- At most `3 * 10^4` calls in total",
    "description_zh": "设计一个栈，除了常规操作外，还能在常数时间内给出栈中的最小元素。实现 `MinStack` 类：\n\n- `MinStack()`：创建一个空栈\n- `push(val)`：把 `val` 压入栈顶\n- `pop()`：删除栈顶元素\n- `top()`：返回栈顶元素\n- `getMin()`：返回栈中当前最小的元素\n\n每个方法的时间复杂度都必须是 **O(1)**。\n\n**示例：** `push(4)`、`push(1)`、`push(6)`、`getMin()` → `1`，`pop()`，`top()` → `1`，`pop()`，`getMin()` → `4`\n\n**约束：**\n- `-2^31 <= val <= 2^31 - 1`\n- 只会在栈非空时调用 `pop`、`top` 和 `getMin`\n- 总调用次数不超过 `3 * 10^4`",
    "hint": "Next to each element, store the minimum of the stack at the moment it was pushed. Popping an element then restores the previous minimum for free.",
    "hint_zh": "每压入一个元素，同时记下压入那一刻整个栈的最小值。这样弹出元素时，之前的最小值自然就恢复了。",
    "cases": [
        {
            "ops": ["MinStack", "push", "push", "push", "getMin", "pop", "top", "pop", "getMin"],
            "args": [[], [4], [1], [6], [], [], [], [], []],
            "expected": [None, None, None, None, 1, None, 1, None, 4],
        },
        {
            "ops": ["MinStack", "push", "push", "getMin", "top", "pop", "getMin"],
            "args": [[], [-3], [5], [], [], [], []],
            "expected": [None, None, None, -3, 5, None, -3],
        },
        {
            # Two copies of the minimum: popping one must not forget the other
            "ops": ["MinStack", "push", "push", "push", "pop", "getMin", "pop", "getMin"],
            "args": [[], [2], [0], [0], [], [], [], []],
            "expected": [None, None, None, None, None, 0, None, 2],
        },
        {
            # The minimum is popped, then the stack refills with larger values
            "ops": ["MinStack", "push", "pop", "push", "push", "getMin", "top"],
            "args": [[], [-7], [], [9], [3], [], []],
            "expected": [None, None, None, None, None, 3, 3],
        },
        {
            "ops": ["MinStack", "push", "push", "getMin", "pop", "getMin"],
            "args": [[], [2147483647], [-2147483648], [], [], []],
            "expected": [None, None, None, -2147483648, None, 2147483647],
        },
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class MinStack:

    def __init__(self):
        self.stack = []  # (value, minimum of the stack up to and including it)

    def push(self, val: int) -> None:
        low = min(val, self.stack[-1][1]) if self.stack else val
        self.stack.append((val, low))

    def pop(self) -> None:
        self.stack.pop()

    def top(self) -> int:
        return self.stack[-1][0]

    def getMin(self) -> int:
        return self.stack[-1][1]''',
    "explanation": "Store pairs `(value, running minimum)`. When a value is pushed, the minimum of the stack becomes `min(value, previous minimum)`, which only depends on the pair below it. Popping removes the pair, and the pair underneath still holds the correct minimum for the smaller stack, so nothing needs recomputing.\n\nTime O(1) per operation, space O(n). A second stack that only records new minimums (pushing on `<=`, so duplicates are kept) saves memory when the minimum rarely changes.",
    "explanation_zh": "栈里存 `(值, 当前最小值)` 二元组。压入一个值时，栈的最小值变为 `min(值, 之前的最小值)`，只依赖它下面那一个二元组。弹出时移除这个二元组，下面的二元组记录的仍是更小的栈的正确最小值，无需重新计算。\n\n每次操作时间 O(1)，空间 O(n)。也可以用一个辅助栈只记录新的最小值（在 `<=` 时压入，以保留重复的最小值），在最小值很少变化时更省内存。",
}
