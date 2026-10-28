# Mistake: merge stops when one list runs out and drops the nodes still left in the other
class Solution:
    def sortList(self, head):
        if head is None or head.next is None:
            return head
        slow, fast = head, head.next
        while fast and fast.next:
            slow, fast = slow.next, fast.next.next
        right, slow.next = slow.next, None
        return self.merge(self.sortList(head), self.sortList(right))

    def merge(self, a, b):
        dummy = tail = ListNode()
        while a and b:
            if a.val <= b.val:
                tail.next, a = a, a.next
            else:
                tail.next, b = b, b.next
            tail = tail.next
        return dummy.next
