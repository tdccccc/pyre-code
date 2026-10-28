# Mistake: no final flush, so bars still on the stack at the end of the array are never measured
class Solution:
    def largestRectangleArea(self, heights):
        stack, best = [], 0
        for i, h in enumerate(heights):
            while stack and heights[stack[-1]] >= h:
                top = stack.pop()
                left = stack[-1] + 1 if stack else 0
                best = max(best, heights[top] * (i - left))
            stack.append(i)
        return best
