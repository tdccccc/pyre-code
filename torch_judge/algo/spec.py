"""Everything derived from a Hot 100 task spec: cases per mode, starters, ACM format text,
the ACM reference program, and display strings.

A task declares its signature once (``params`` / ``returns`` or ``design``) plus plain-data
cases; both answer modes are generated from that, so a problem is written exactly once.
"""

from __future__ import annotations

import functools
import random

from torch_judge.algo import runner

# LeetCode Hot 100 sections, in study-plan order
CATEGORIES: list[tuple[str, str]] = [
    ("Hashing", "哈希"),
    ("Two Pointers", "双指针"),
    ("Sliding Window", "滑动窗口"),
    ("Substring", "子串"),
    ("Arrays", "普通数组"),
    ("Matrix", "矩阵"),
    ("Linked List", "链表"),
    ("Binary Tree", "二叉树"),
    ("Graphs", "图论"),
    ("Backtracking", "回溯"),
    ("Binary Search", "二分查找"),
    ("Stack", "栈"),
    ("Heap", "堆"),
    ("Greedy", "贪心算法"),
    ("Dynamic Programming", "动态规划"),
    ("Multi-dimensional DP", "多维动态规划"),
    ("Techniques", "技巧"),
]
CATEGORY_ZH = dict(CATEGORIES)
CATEGORY_INDEX = {name: i for i, (name, _) in enumerate(CATEGORIES)}

MODES = ("core", "acm")

_PY_HINTS = {
    "int": "int", "float": "float", "bool": "bool", "str": "str",
    "int[]": "List[int]", "int[][]": "List[List[int]]",
    "str[]": "List[str]", "str[][]": "List[List[str]]", "char[][]": "List[List[str]]",
    "ListNode": "Optional[ListNode]", "ListNode[]": "List[Optional[ListNode]]",
    "TreeNode": "Optional[TreeNode]", "Node": "Optional[Node]",
    "void": "None",
}

LISTNODE_COMMENT = """# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
"""

TREENODE_COMMENT = """# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
"""

NODE_COMMENT = """# Definition for a Node.
# class Node:
#     def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
#         self.val = int(x)
#         self.next = next
#         self.random = random
"""


def sort_key(task: dict) -> tuple[int, int]:
    return CATEGORY_INDEX[task["category"]], task["leetcode_id"]


def _node_kinds(types: set[str]) -> set[str]:
    """ListNode / TreeNode / Node among the given type names (``ListNode[]`` counts as ListNode)."""
    return {kind for kind in ("ListNode", "TreeNode", "Node") if any(t.rstrip("[]") == kind for t in types)}


def _signature_types(task: dict) -> set[str]:
    if task.get("design"):
        design = task["design"]
        types = set(design["init"].values())
        for params, ret in design["methods"].values():
            types |= set(params.values()) | {ret}
        return types
    return {t for _, t in task["params"]} | {task["returns"]}


def _types_used(task: dict) -> set[str]:
    if task.get("design"):
        return _signature_types(task)
    return _signature_types(task) | {t for _, t in inputs(task)} | {acm_output_type(task)}


def inputs(task: dict) -> list:
    """The plain data a case's ``args`` hold, as (name, type). Same as ``params`` unless the
    task builds the method's arguments itself with a ``prepare`` hook."""
    return task.get("inputs") or task["params"]


def _spec(task: dict) -> dict:
    """The subset of the task the core-mode runner needs."""
    if task.get("design"):
        design = task["design"]
        returns_float = any(ret == "float" for _, ret in design["methods"].values())
        return {"design": {
            "class": design["class"],
            "methods": {name: [list(params.values()), ret] for name, (params, ret) in design["methods"].items()},
        }, "check": "approx" if returns_float else "exact"}
    return {
        "function_name": task["function_name"],
        "params": [list(p) for p in task["params"]],
        "inputs": [list(p) for p in inputs(task)],
        "returns": task["returns"],
        "inplace": task.get("inplace"),
        "prepare": task.get("prepare"),
        "finish": task.get("finish"),
        "check": task.get("check", "exact"),
        "checker": task.get("checker"),
    }


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

RANDOM_CASES = 20


@functools.lru_cache(maxsize=None)
def _large_case(task_id: str, factory) -> dict:
    return factory()


@functools.lru_cache(maxsize=None)
def _random_cases(task_id: str) -> tuple[dict, ...]:
    """Small random cases from the task's ``random_case(rng)``, answered by the reference
    solution. Seeded by task id, so every run (and the exported UI data) sees the same ones."""
    from torch_judge.tasks import get_task

    task = get_task(task_id)
    rng = random.Random(f"{task_id}:random")
    ns = runner.solution_namespace("__reference__")
    exec(task["solution"], ns)
    cases = []
    for i in range(RANDOM_CASES):
        case = task["random_case"](rng)
        case["expected"] = runner.compute(ns, _spec(task), case)
        case["name"] = f"Random {i + 1}"
        cases.append(case)
    return tuple(cases)


def cases_for_mode(task_id: str, task: dict, mode: str) -> list[dict]:
    """Ordered cases for a mode: hand-written (the first two are the samples Run uses),
    then seeded random ones, then the hidden large input."""
    cases = [dict(c) for c in task["cases"] if mode == "core" or c.get("acm", True)]
    for i, case in enumerate(cases):
        case.setdefault("name", f"Example {i + 1}" if i < 2 else f"Case {i + 1}")
    if task.get("random_case"):
        cases.extend(dict(c) for c in _random_cases(task_id))
    if task.get("large_case"):
        large = dict(_large_case(task_id, task["large_case"]))
        large.update(name="Large input", hidden=True)
        cases.append(large)
    return cases


def acm_stdin(task: dict, case: dict) -> str:
    if task.get("design"):
        lines = [str(len(case["ops"]))]
        for op, args in zip(case["ops"], case["args"]):
            lines.append(" ".join([op] + [runner._scalar_text(a) for a in args]))
        return "\n".join(lines) + "\n"
    lines: list[str] = []
    for (_, type_name), arg in zip(inputs(task), case["args"]):
        lines.extend(runner.acm_input_lines(type_name, arg))
    return "\n".join(lines) + "\n"


def acm_output_type(task: dict) -> str:
    """Type of the plain answer: ``output`` when a ``finish`` hook converts the result."""
    if task.get("output"):
        return task["output"]
    if task.get("inplace"):
        return dict(task["params"])[task["inplace"]]
    return task["returns"]


def acm_expected_stdout(task: dict, case: dict) -> str:
    if task.get("design"):
        return "\n".join(runner._scalar_text(v) for v in _design_outputs(task, case))
    return runner.acm_output_text(acm_output_type(task), case["expected"])


def _design_outputs(task: dict, case: dict) -> list:
    return [value for _, value in design_outputs(task, case)]


def design_outputs(task: dict, case: dict) -> list[tuple[str, object]]:
    """(return type, expected value) of every non-void call, in order."""
    methods = task["design"]["methods"]
    return [
        (methods[op][1], value) for op, value in zip(case["ops"][1:], case["expected"][1:])
        if methods[op][1] != "void"
    ]


def display_input(task: dict, case: dict, mode: str) -> str:
    if mode == "acm":
        return runner._clip(acm_stdin(task, case).rstrip("\n"))
    if task.get("design"):
        return runner._clip(f"{runner.show(case['ops'])}\n{runner.show(case['args'])}")
    return runner.show_args(inputs(task), case["args"])


def display_expected(task: dict, case: dict, mode: str) -> str:
    if mode == "acm":
        return runner._clip(acm_expected_stdout(task, case))
    return runner.show(case["expected"])


# ---------------------------------------------------------------------------
# Starters
# ---------------------------------------------------------------------------

def _signature(name: str, params: dict | list, ret: str | None) -> str:
    items = params.items() if isinstance(params, dict) else params
    args = "".join(f", {p}: {_PY_HINTS[t]}" for p, t in items)
    return f"def {name}(self{args}){f' -> {_PY_HINTS[ret]}' if ret else ''}:"


def core_starter(task: dict) -> str:
    kinds = _node_kinds(_signature_types(task))
    parts = ["from typing import List, Optional\n"]
    if "ListNode" in kinds:
        parts.append(LISTNODE_COMMENT)
    if "TreeNode" in kinds:
        parts.append(TREENODE_COMMENT)
    if "Node" in kinds:
        parts.append(NODE_COMMENT)
    if task.get("design"):
        design = task["design"]
        body = [f"class {design['class']}:", "", f"    {_signature('__init__', design['init'], None)}", "        pass"]
        for name, (params, ret) in design["methods"].items():
            body += ["", f"    {_signature(name, params, ret)}", "        pass"]
        parts.append("\n".join(body) + "\n")
    else:
        returns = "void" if task.get("inplace") else task["returns"]
        sig = _signature(task["function_name"], task["params"], returns)
        parts.append(f"class Solution:\n    {sig}\n        pass\n")
    return "\n".join(parts)


ACM_STARTER = '''import sys


def main():
    data = sys.stdin.read().split("\\n")
    # Parse the input as described under "Input format", then print the answer.


if __name__ == "__main__":
    main()
'''


# ---------------------------------------------------------------------------
# ACM format description
# ---------------------------------------------------------------------------

def _input_lines(name: str, type_name: str) -> list[tuple[str, str, bool]]:
    """(en, zh, fixed_height) descriptions of the lines one argument occupies."""
    if type_name == "int":
        return [(f"an integer `{name}`", f"整数 `{name}`", True)]
    if type_name == "bool":
        return [(f"`{name}`: `true` or `false`", f"`{name}`，取值为 `true` 或 `false`", True)]
    if type_name == "str":
        return [(f"the string `{name}` (the whole line, possibly empty)", f"字符串 `{name}`（整行内容，可能为空）", True)]
    if type_name == "int[]":
        return [
            (f"an integer n, the length of `{name}`", f"整数 n，表示 `{name}` 的长度", True),
            (f"n space-separated integers: `{name}`", f"n 个以空格分隔的整数，即 `{name}`", True),
        ]
    if type_name == "ListNode":
        return [
            (f"an integer n, the number of nodes in `{name}`", f"整数 n，表示链表 `{name}` 的节点数", True),
            (f"n space-separated integers: the node values from head to tail", "n 个以空格分隔的整数，即从头到尾各节点的值", True),
        ]
    if type_name == "int[][]":
        return [
            (f"an integer m, the number of rows of `{name}`", f"整数 m，表示 `{name}` 的行数", True),
            (f"the next m lines: one row of `{name}` each, space-separated", f"接下来 m 行：每行是 `{name}` 的一行，以空格分隔", False),
        ]
    if type_name == "char[][]":
        return [
            (f"an integer m, the number of rows of `{name}`", f"整数 m，表示 `{name}` 的行数", True),
            (f"the next m lines: one row of `{name}` each, its characters written together without spaces",
             f"接下来 m 行：每行是 `{name}` 的一行，字符之间没有空格", False),
        ]
    if type_name == "str[]":
        return [
            (f"an integer n, the length of `{name}`", f"整数 n，表示 `{name}` 的长度", True),
            (f"the next n lines: one string of `{name}` each", f"接下来 n 行：每行是 `{name}` 中的一个字符串", False),
        ]
    if type_name == "ListNode[]":
        return [
            (f"an integer k, the number of linked lists in `{name}`", f"整数 k，表示 `{name}` 中链表的个数", True),
            ("the next 2k lines: for each list, its node count n on one line and its n values, head to tail, on the next",
             "接下来 2k 行：每个链表占两行，第一行是节点数 n，第二行是从头到尾的 n 个值，以空格分隔", False),
        ]
    if type_name == "Node":
        return [
            (f"an integer n, the number of nodes in `{name}`", f"整数 n，表示链表 `{name}` 的节点数", True),
            ("the next n lines: one node each, head to tail: its value and the index its random pointer points to (`null` for none)",
             "接下来 n 行：从头到尾每行一个节点，写出它的值和 random 指针指向的节点下标（为空写 `null`）", False),
        ]
    if type_name == "TreeNode":
        return [(
            f"the level-order traversal of `{name}`, space-separated, `null` for missing nodes (empty line for an empty tree)",
            f"二叉树 `{name}` 的层序遍历，以空格分隔，空节点写作 `null`（空树为空行）",
            True,
        )]
    raise NotImplementedError(type_name)


def _number_lines(entries: list[tuple[str, str, bool]]) -> tuple[list[str], list[str]]:
    en, zh = [], []
    line_no, fixed = 1, True
    for desc_en, desc_zh, height_fixed in entries:
        if desc_zh.startswith("接下来"):
            en.append(f"- {desc_en[0].upper()}{desc_en[1:]}")
            zh.append(f"- {desc_zh}")
        elif fixed:
            en.append(f"- Line {line_no}: {desc_en}")
            zh.append(f"- 第 {line_no} 行：{desc_zh}")
        else:
            en.append(f"- Next line: {desc_en}")
            zh.append(f"- 下一行：{desc_zh}")
        line_no += 1
        fixed = fixed and height_fixed
    return en, zh


_OUTPUT_DESC = {
    "int": ("one integer", "一个整数"),
    "float": ("one number", "一个数"),
    "bool": ("`true` or `false`", "`true` 或 `false`"),
    "str": ("one line: the string", "一行字符串"),
    "int[]": ("one line of space-separated integers (an empty line if the result is empty)", "一行以空格分隔的整数（结果为空时输出空行）"),
    "ListNode": ("one line: the node values from head to tail, space-separated (an empty line for an empty list)", "一行：从头到尾各节点的值，以空格分隔（空链表输出空行）"),
    "TreeNode": ("one line: the level-order traversal, `null` for missing nodes", "一行：层序遍历，空节点写作 `null`"),
    "int[][]": ("one inner list per line, space-separated", "若干行，每行一个子列表，元素以空格分隔"),
    "str[][]": ("one group per line, strings space-separated", "若干行，每行一组，字符串以空格分隔"),
    "str[]": ("one string per line", "若干行，每行一个字符串"),
    "char[][]": ("one row per line, characters written together without spaces", "若干行，每行是网格的一行，字符之间没有空格"),
    "Node": ("one node per line, head to tail: its value and the index its random pointer points to (`null` for none)",
             "若干行，从头到尾每行一个节点：它的值和 random 指针指向的节点下标（为空写 `null`）"),
}

_CHECK_NOTE = {
    "unordered": ("Elements may be printed in any order.", "元素可以按任意顺序输出。"),
    "unordered_nested": ("Groups, and the elements within a group, may be in any order.", "组与组之间、组内元素的顺序均不限。"),
    "custom": ("If several answers are valid, print any one of them.", "答案不唯一时，输出任意一个正确答案即可。"),
    "float": ("Answers within 1e-5 of the expected value are accepted.", "与标准答案相差不超过 1e-5 即视为正确。"),
}


def acm_format(task: dict) -> dict[str, str]:
    if task.get("design"):
        design = task["design"]
        cls = design["class"]
        construct = " ".join([cls, *design["init"]])
        returning = [name for name, (_, ret) in design["methods"].items() if ret != "void"]
        ops = ", ".join(f"`{' '.join([n, *p])}`" for n, (p, _) in design["methods"].items())
        input_en = "\n".join([
            "- Line 1: an integer q, the number of operations (including construction)",
            "- The next q lines: one operation each, the name followed by its arguments, space-separated",
            f"- The first operation is always `{construct}`; the rest are among {ops}",
        ])
        input_zh = "\n".join([
            "- 第 1 行：整数 q，表示操作数（包含构造）",
            "- 接下来 q 行：每行一个操作，先写操作名，再写参数，以空格分隔",
            f"- 第一个操作总是 `{construct}`，其余操作为 {ops} 之一",
        ])
        names = ", ".join(f"`{n}`" for n in returning)
        output_en = f"For every {names} call, in order, print its return value on its own line."
        output_zh = f"对每次 {names} 调用，按顺序各输出一行返回值。"
        if any(ret == "bool" for _, ret in design["methods"].values()):
            output_en += " Print booleans as `true` / `false`."
            output_zh += "布尔值输出 `true` / `false`。"
        if any(ret == "float" for _, ret in design["methods"].values()):
            output_en += f" {_CHECK_NOTE['float'][0]}"
            output_zh += _CHECK_NOTE["float"][1]
        return {"inputEn": input_en, "inputZh": input_zh, "outputEn": output_en, "outputZh": output_zh}

    entries = [line for name, t in inputs(task) for line in _input_lines(name, t)]
    input_en, input_zh = _number_lines(entries)
    out_type = acm_output_type(task)
    out_en, out_zh = _OUTPUT_DESC[out_type.rstrip("?")]
    if out_type.endswith("?"):
        out_en, out_zh = f"{out_en}, or `null` if there is none", f"{out_zh}；不存在时输出 `null`"
    if task.get("inplace"):
        out_en = f"`{task['inplace']}` after the operation, as {out_en}"
        out_zh = f"操作完成后的 `{task['inplace']}`，格式为{out_zh}"
    out_en, out_zh = f"Print {out_en}.", f"输出{out_zh}。"
    note = _CHECK_NOTE.get(task.get("check", "exact"))
    if note:
        out_en, out_zh = f"{out_en} {note[0]}", f"{out_zh}{note[1]}"
    return {
        "inputEn": "\n".join(input_en),
        "inputZh": "\n".join(input_zh),
        "outputEn": out_en,
        "outputZh": out_zh,
    }


# ---------------------------------------------------------------------------
# ACM reference program (core solution + I/O glue)
# ---------------------------------------------------------------------------

_HELPERS = {
    "ListNode": '''class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


def build_list(values):
    dummy = tail = ListNode()
    for v in values:
        tail.next = ListNode(v)
        tail = tail.next
    return dummy.next


def list_to_array(head):
    out = []
    while head:
        out.append(head.val)
        head = head.next
    return out
''',
    "TreeNode": '''from collections import deque


class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


def build_tree(values):
    """Level-order values, None for a missing node."""
    if not values or values[0] is None:
        return None
    root = TreeNode(values[0])
    queue, i = deque([root]), 1
    while queue and i < len(values):
        node = queue.popleft()
        if i < len(values) and values[i] is not None:
            node.left = TreeNode(values[i])
            queue.append(node.left)
        i += 1
        if i < len(values) and values[i] is not None:
            node.right = TreeNode(values[i])
            queue.append(node.right)
        i += 1
    return root


def tree_to_array(root):
    out, queue = [], deque([root])
    while queue:
        node = queue.popleft()
        out.append(node.val if node else None)
        if node:
            queue.extend([node.left, node.right])
    while out and out[-1] is None:
        out.pop()
    return out
''',
    "Node": '''class Node:
    def __init__(self, x, next=None, random=None):
        self.val = int(x)
        self.next = next
        self.random = random


def build_random_list(pairs):
    nodes = [Node(val) for val, _ in pairs]
    for i, (_, rand) in enumerate(pairs):
        if i + 1 < len(nodes):
            nodes[i].next = nodes[i + 1]
        nodes[i].random = None if rand is None else nodes[rand]
    return nodes[0] if nodes else None


def random_list_to_array(head):
    nodes = []
    while head:
        nodes.append(head)
        head = head.next
    index = {id(n): i for i, n in enumerate(nodes)}
    return [[n.val, None if n.random is None else index[id(n.random)]] for n in nodes]
''',
    # Lines with `null` tokens (trees, random-pointer lists)
    "tokens": '''def parse_tokens(line):
    return [None if t == "null" else int(t) for t in line.split()]


def format_tokens(values):
    return " ".join("null" if v is None else str(v) for v in values)
''',
}


def _read_stmt(name: str, type_name: str) -> list[str]:
    if type_name == "int":
        return [f"{name} = int(next(lines))"]
    if type_name == "float":
        return [f"{name} = float(next(lines))"]
    if type_name == "bool":
        return [f'{name} = next(lines).strip() == "true"']
    if type_name == "str":
        return [f"{name} = next(lines)"]
    if type_name == "int[]":
        return [f"next(lines)  # length of {name}", f"{name} = list(map(int, next(lines).split()))"]
    if type_name == "ListNode":
        return [f"next(lines)  # number of nodes", f"{name} = build_list(list(map(int, next(lines).split())))"]
    if type_name == "int[][]":
        return ["m = int(next(lines))", f"{name} = [list(map(int, next(lines).split())) for _ in range(m)]"]
    if type_name == "char[][]":
        return ["m = int(next(lines))", f"{name} = [list(next(lines).strip()) for _ in range(m)]"]
    if type_name == "str[]":
        return ["n = int(next(lines))", f"{name} = [next(lines) for _ in range(n)]"]
    if type_name == "ListNode[]":
        return [
            "k = int(next(lines))",
            f"{name} = []",
            "for _ in range(k):",
            "    next(lines)  # number of nodes",
            f"    {name}.append(build_list(list(map(int, next(lines).split()))))",
        ]
    if type_name == "TreeNode":
        return [f"{name} = build_tree(parse_tokens(next(lines)))"]
    if type_name == "Node":
        return ["n = int(next(lines))", f"{name} = build_random_list([parse_tokens(next(lines)) for _ in range(n)])"]
    raise NotImplementedError(type_name)


def _print_stmt(expr: str, type_name: str) -> list[str]:
    if type_name.endswith("?"):
        return [f'print("null" if {expr} is None else {expr})']
    if type_name in ("int", "float", "str"):
        return [f"print({expr})"]
    if type_name == "bool":
        return [f'print("true" if {expr} else "false")']
    if type_name == "int[]":
        return [f'print(" ".join(map(str, {expr})))']
    if type_name == "ListNode":
        return [f'print(" ".join(map(str, list_to_array({expr}))))']
    if type_name == "TreeNode":
        return [f"print(format_tokens(tree_to_array({expr})))"]
    if type_name in ("int[][]", "str[][]"):
        return [f"for row in {expr}:", '    print(" ".join(map(str, row)))']
    if type_name == "char[][]":
        return [f"for row in {expr}:", '    print("".join(row))']
    if type_name == "Node":
        return [f"for row in random_list_to_array({expr}):", "    print(format_tokens(row))"]
    if type_name == "str[]":
        return [f"for item in {expr}:", "    print(item)"]
    raise NotImplementedError(type_name)


_CAST = {"int": "int({})", "float": "float({})", "str": "{}", "bool": '{} == "true"'}


def acm_solution(task: dict) -> str:
    kinds = _node_kinds(_types_used(task))
    if kinds & {"TreeNode", "Node"}:
        kinds.add("tokens")
    header = ["import sys", "from typing import List, Optional", ""]
    helpers = [_HELPERS[t] for t in ("ListNode", "TreeNode", "Node", "tokens") if t in kinds]
    # Hooks use the same helper names as in the runner, so they run here unchanged
    hooks = [task[k].strip() + "\n" for k in ("prepare", "finish") if task.get(k)]
    core = task["solution"].strip()

    body = ["def main():", '    lines = iter(sys.stdin.read().split("\\n"))']
    if task.get("design"):
        design = task["design"]
        body += ["    q = int(next(lines))", "    obj = None", "    for _ in range(q):", "        op, *args = next(lines).split()"]
        branches = [(design["class"], list(design["init"].values()), None)]
        branches += [(name, list(params.values()), ret) for name, (params, ret) in design["methods"].items()]
        for i, (name, param_types, ret) in enumerate(branches):
            call_args = ", ".join(_CAST[t].format(f"args[{j}]") for j, t in enumerate(param_types))
            body.append(f'        {"if" if i == 0 else "elif"} op == "{name}":')
            if ret is None:
                body.append(f"            obj = {name}({call_args})")
            elif ret == "void":
                body.append(f"            obj.{name}({call_args})")
            else:
                body += [f"            {s}" for s in _print_stmt(f"obj.{name}({call_args})", ret)]
    else:
        for name, type_name in inputs(task):
            body += [f"    {s}" for s in _read_stmt(name, type_name)]
        names = ", ".join(n for n, _ in inputs(task))
        method = f"Solution().{task['function_name']}"
        if task.get("prepare"):
            body.append(f"    args = prepare([{names}])")
            call = f"{method}(*args)"
        else:
            if task.get("finish"):
                body.append(f"    args = [{names}]")
            call = f"{method}({names})"
        if task.get("inplace"):
            body.append(f"    {call}")
            body += [f"    {s}" for s in _print_stmt(task["inplace"], acm_output_type(task))]
        else:
            body.append(f"    ans = {call}")
            if task.get("finish"):
                body.append("    ans = finish(ans, args)")
            body += [f"    {s}" for s in _print_stmt("ans", acm_output_type(task))]

    footer = ["", "", 'if __name__ == "__main__":', "    main()", ""]
    sections = ["\n".join(header)] + helpers + hooks + [core, "\n".join(body)]
    return "\n\n".join(s.rstrip("\n") + "\n" for s in sections).rstrip("\n") + "\n" + "\n".join(footer)


def runner_spec(task: dict) -> dict:
    return _spec(task)
