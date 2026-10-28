# Mistake: moves the taller wall inward, so the shorter wall that limits the area is discarded
class Solution:
    def maxArea(self, height):
        lo, hi = 0, len(height) - 1
        best = 0
        while lo < hi:
            best = max(best, min(height[lo], height[hi]) * (hi - lo))
            if height[lo] < height[hi]:
                hi -= 1
            else:
                lo += 1
        return best
