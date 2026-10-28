# Mistake: compares the first half with the second half in the same direction, forgetting to reverse one of them
class Solution:
    def isPalindrome(self, head):
        vals = []
        while head:
            vals.append(head.val)
            head = head.next
        half = len(vals) // 2
        return vals[:half] == vals[len(vals) - half:]
