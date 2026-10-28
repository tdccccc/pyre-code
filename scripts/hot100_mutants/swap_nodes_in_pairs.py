# Mistake: recursive version forgets to point the second node back at the first, so nodes are dropped
class Solution:
    def swapPairs(self, head):
        if not head or not head.next:
            return head
        second = head.next
        head.next = self.swapPairs(second.next)
        return head
