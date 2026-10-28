# Mistake: recursive reversal that never clears the old head's next pointer, leaving a cycle
class Solution:
    def reverseList(self, head):
        if head is None or head.next is None:
            return head
        new_head = self.reverseList(head.next)
        head.next.next = head
        return new_head
