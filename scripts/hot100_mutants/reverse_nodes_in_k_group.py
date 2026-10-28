# Mistake: reverses the last group even when it has fewer than k nodes
class Solution:
    def reverseKGroup(self, head, k):
        dummy = ListNode(0, head)
        group_prev = dummy
        while group_prev.next:
            kth = group_prev
            for _ in range(k):
                if kth.next is None:
                    break
                kth = kth.next
            group_next = kth.next
            prev, cur = group_next, group_prev.next
            while cur is not group_next:
                cur.next, prev, cur = prev, cur, cur.next
            first = group_prev.next
            group_prev.next = prev
            group_prev = first
        return dummy.next
