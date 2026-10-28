# Mistake: returns where the fast and slow pointers meet, which is inside the cycle but not necessarily at its start
class Solution:
    def detectCycle(self, head):
        slow = fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if slow is fast:
                return slow
        return None
