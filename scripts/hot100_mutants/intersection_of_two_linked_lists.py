# Mistake: compares node values instead of node identity
class Solution:
    def getIntersectionNode(self, headA, headB):
        a, b = headA, headB
        while a and b:
            if a.val == b.val:
                return a
            a, b = a.next, b.next
        return None
