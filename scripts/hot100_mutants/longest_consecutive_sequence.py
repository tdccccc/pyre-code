# Mistake: sorts, then restarts the run on a repeated value, so duplicates break a streak
class Solution:
    def longestConsecutive(self, nums):
        best = cur = 0
        prev = None
        for x in sorted(nums):
            cur = cur + 1 if prev is not None and x == prev + 1 else 1
            best = max(best, cur)
            prev = x
        return best
