# Mistake: keeps only the running maximum, so a negative times a negative is never seen
class Solution:
    def maxProduct(self, nums):
        best = cur = nums[0]
        for x in nums[1:]:
            cur = max(x, cur * x)
            best = max(best, cur)
        return best
