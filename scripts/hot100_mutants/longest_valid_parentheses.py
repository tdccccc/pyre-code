# Mistake: an unmatched ')' is never recorded as the new base, so the old base stays too far left
class Solution:
    def longestValidParentheses(self, s):
        best = 0
        stack = [-1]
        for i, ch in enumerate(s):
            if ch == "(":
                stack.append(i)
            elif len(stack) > 1:
                stack.pop()
                best = max(best, i - stack[-1])
        return best
