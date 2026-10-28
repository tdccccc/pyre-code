"""Add Two Numbers (LeetCode 2)."""

import random


def _to_int(digits: list) -> int:
    """Digits stored least significant first."""
    return sum(d * 10**i for i, d in enumerate(digits))


def _to_digits(number: int) -> list:
    return [int(c) for c in str(number)[::-1]]


def _large_case():
    # Two 100-digit numbers (the constraint), both with a 9 on top, so the sum carries out of
    # the top digit and needs a 101st node. Nothing can time out at this size; the case checks
    # the carry and the unequal-length handling at full length.
    rng = random.Random(2)
    a = [rng.randint(0, 9) for _ in range(99)] + [9]
    b = [rng.randint(0, 9) for _ in range(99)] + [9]
    return {"args": [a, b], "expected": _to_digits(_to_int(a) + _to_int(b))}


def _random_case(rng):
    def number():
        n = rng.randint(1, 4)
        return [rng.randint(0, 9) for _ in range(n - 1)] + [rng.randint(1, 9)]

    return {"args": [number(), number()]}


TASK = {
    "title": "Add Two Numbers",
    "title_zh": "两数相加",
    "difficulty": "Medium",
    "leetcode_id": 2,
    "leetcode_slug": "add-two-numbers",
    "category": "Linked List",
    "function_name": "addTwoNumbers",
    "params": [("l1", "ListNode"), ("l2", "ListNode")],
    "returns": "ListNode",
    "description_en": "You are given two non-empty linked lists that represent two non-negative integers. The digits are stored in reverse order, so the head holds the least significant digit, and each node holds one digit. Add the two numbers and return the sum as a linked list in the same format.\n\n**Example 1:** `l1 = [3,1,5], l2 = [8,4]` → `[1,6,5]` (513 + 48 = 561)\n\n**Example 2:** `l1 = [9,9], l2 = [1]` → `[0,0,1]` (99 + 1 = 100)\n\n**Constraints:**\n- Each list has `1` to `100` nodes\n- $0 \\le \\texttt{Node.val} \\le 9$\n- No list has a leading zero, except the number `0` itself\n\n**Follow-up:** how would you change the method if the digits were stored most significant first?",
    "description_zh": "给定两个非空链表，分别表示两个非负整数。每个链表的数字按逆序存储：头节点是个位，每个节点存放一位数字。把两个数相加，并以同样的链表形式返回它们的和。\n\n**示例 1：** `l1 = [3,1,5], l2 = [8,4]` → `[1,6,5]`（513 + 48 = 561）\n\n**示例 2：** `l1 = [9,9], l2 = [1]` → `[0,0,1]`（99 + 1 = 100）\n\n**约束：**\n- 每个链表有 `1` 到 `100` 个节点\n- $0 \\le \\texttt{Node.val} \\le 9$\n- 除数字 `0` 本身外，链表都没有前导零\n\n**进阶：** 如果数字改为高位在前存储，你会怎样修改你的方法？",
    "hint": "Add digit by digit from the heads, carrying into the next position. Do not forget a carry left over after both lists have run out.",
    "hint_zh": "从两个链表的头开始逐位相加，把进位带到下一位。两个链表都走完后，别忘了可能还剩一个进位。",
    "cases": [
        {"args": [[3, 1, 5], [8, 4]], "expected": [1, 6, 5]},
        # The final carry becomes a new leading node: 99 + 1 = 100
        {"args": [[9, 9], [1]], "expected": [0, 0, 1]},
        {"args": [[0], [0]], "expected": [0]},
        {"args": [[7, 2, 9], [6, 5]], "expected": [3, 8, 9]},
        {"args": [[5], [5]], "expected": [0, 1]},
        # Both lists end in a carry chain of nines, and the longer list keeps its own digits
        {"args": [[9, 9, 9], [9, 9, 9, 9]], "expected": [8, 9, 9, 0, 1]},
        # The shorter list is padded with zeros
        {"args": [[1, 8], [0]], "expected": [1, 8]},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def addTwoNumbers(self, l1: Optional[ListNode], l2: Optional[ListNode]) -> Optional[ListNode]:
        dummy = tail = ListNode()
        carry = 0
        # Keep going while digits or a carry remain, so a carry out of the top digit still counts
        while l1 or l2 or carry:
            total = carry
            if l1:
                total += l1.val
                l1 = l1.next
            if l2:
                total += l2.val
                l2 = l2.next
            tail.next = ListNode(total % 10)
            tail = tail.next
            carry = total // 10
        return dummy.next''',
    "explanation": "Add the two lists together digit by digit, the way one does by hand. At each position, add the two digits (a missing digit counts as 0) and the carry from the previous position. Append the last digit of that sum as a new node and keep the tens digit as the carry. The loop also runs while a carry remains after both lists are exhausted, which is what produces the leading `1` in 99 + 1.\n\nTime O(max(m, n)), space O(max(m, n)) for the output list.",
    "explanation_zh": "像手工竖式一样逐位相加。每一位把两个数位（缺失的数位按 0 计）与上一位的进位相加，结果的个位接成新节点，十位作为新的进位。循环条件里保留了「进位未处理」的情况，这正是 99 + 1 得到前导 `1` 的原因。\n\n时间 O(max(m, n))，空间 O(max(m, n))（输出链表）。",
}
