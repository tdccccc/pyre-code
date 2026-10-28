"""Palindrome Linked List (LeetCode 234)."""

import random


def _is_palindrome(values: list) -> bool:
    return all(values[i] == values[-1 - i] for i in range(len(values) // 2))


def _large_case():
    # 10^5 nodes whose mirror pair next to the centre is broken, so the answer is False and
    # the mismatch sits as deep in the list as possible. A recursive comparison needs one
    # frame per node and overflows the runner's recursion limit (20000) at this size.
    rng = random.Random(234)
    n = 10**5
    half = [rng.randint(0, 9) for _ in range(n // 2)]
    values = half + half[::-1]
    values[n // 2 - 1] = (values[n // 2 - 1] + 1) % 10
    return {"args": [values], "expected": _is_palindrome(values)}


def _palindrome(rng, n: int) -> list:
    half = [rng.randint(0, 2) for _ in range((n + 1) // 2)]
    return half + half[: n // 2][::-1]


def _random_case(rng):
    # The mistake compares the halves without reversing one. It wrongly rejects palindromes
    # whose first half is not itself a palindrome (1 2 2 1), and wrongly accepts block repeats
    # like 1 2 1 2 (equal halves, not a palindrome). Measured: about 10 of 20 cases expose it
    r = rng.random()
    if r < 0.5:
        return {"args": [_palindrome(rng, rng.randint(4, 8))]}
    if r < 0.8:
        block = [rng.randint(0, 2) for _ in range(rng.randint(1, 3))]
        return {"args": [block + block]}
    return {"args": [[rng.randint(0, 2) for _ in range(rng.randint(2, 8))]]}


TASK = {
    "title": "Palindrome Linked List",
    "title_zh": "回文链表",
    "difficulty": "Easy",
    "leetcode_id": 234,
    "leetcode_slug": "palindrome-linked-list",
    "category": "Linked List",
    "function_name": "isPalindrome",
    "params": [("head", "ListNode")],
    "returns": "bool",
    "description_en": "Given the head of a singly linked list, decide whether its values read the same forwards and backwards. Return `true` if they do, otherwise `false`.\n\n**Example 1:** `head = [6,1,6]` → `true`\n\n**Example 2:** `head = [9,4,4,7]` → `false`\n\n**Constraints:**\n- The list has between `1` and $10^{5}$ nodes\n- $0 \\le \\texttt{Node.val} \\le 9$\n\n**Follow-up:** can you check it without copying the values into an array, using constant extra memory?",
    "description_zh": "给定单链表的头节点 `head`，判断它的值从前往后读和从后往前读是否相同。相同返回 `true`，否则返回 `false`。\n\n**示例 1：** `head = [6,1,6]` → `true`\n\n**示例 2：** `head = [9,4,4,7]` → `false`\n\n**约束：**\n- 链表节点数在 `1` 到 $10^{5}$ 之间\n- $0 \\le \\texttt{Node.val} \\le 9$\n\n**进阶：** 你能不把值复制到数组里，只用常数额外内存完成判断吗？",
    "hint": "Find the middle with a slow and a fast pointer, reverse the second half, then compare the two halves node by node.",
    "hint_zh": "用快慢指针找中点，反转后半段，然后逐个节点比较两半。",
    "cases": [
        {"args": [[6, 1, 6]], "expected": True},
        {"args": [[9, 4, 4, 7]], "expected": False},
        {"args": [[3]], "expected": True},
        {"args": [[2, 3]], "expected": False},
        {"args": [[5, 5]], "expected": True},
        {"args": [[1, 2, 3, 2, 1]], "expected": True},
        # The first half equals the second half, but not reversed: a check that forgets to reverse fails
        {"args": [[4, 7, 4, 7]], "expected": False},
        # Outer values match, inner values do not: a check that only compares the ends fails
        {"args": [[0, 1, 0, 0]], "expected": False},
    ],
    "random_case": _random_case,
    "large_case": _large_case,
    "solution": '''class Solution:
    def isPalindrome(self, head: Optional[ListNode]) -> bool:
        # Find the middle with a slow and a fast pointer, then reverse the second half in place
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        prev = None
        while slow:
            nxt = slow.next
            slow.next = prev
            prev, slow = slow, nxt
        left, right = head, prev
        while right:
            if left.val != right.val:
                return False
            left, right = left.next, right.next
        return True''',
    "explanation": "Find the middle with two pointers (`fast` moves two steps for each step of `slow`), reverse the second half in place, then walk both halves together, from the two ends inwards. Any mismatch means the list is not a palindrome. For an odd-length list the middle node ends up compared with itself, which is harmless.\n\nTime O(n), space O(1). The reversal rewires the list; reverse the second half back if the caller needs the original.",
    "explanation_zh": "用两个指针找到中点（`fast` 每走两步，`slow` 走一步），原地反转后半部分，然后从两端向中间同时比较。只要有一处不相等，就不是回文。奇数长度时，中间节点最后会和它自己比较，结果不受影响。\n\n时间 O(n)，空间 O(1)。反转会改变链表的指针；如果调用方需要原链表，应把后半部分再反转回来。",
}
