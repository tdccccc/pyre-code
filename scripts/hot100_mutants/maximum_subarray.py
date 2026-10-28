# Mistake: starts the best sum at 0, so an all-negative array wrongly returns 0
class Solution:
    def maxSubArray(self, nums):
        best = cur = 0
        for x in nums:
            cur = max(x, cur + x)
            best = max(best, cur)
        return best
