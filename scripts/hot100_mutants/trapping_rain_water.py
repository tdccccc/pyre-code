# Mistake: each cell's water is taken from its immediate neighbours instead of the tallest walls on each side
class Solution:
    def trap(self, height):
        total = 0
        for i in range(1, len(height) - 1):
            total += max(0, min(height[i - 1], height[i + 1]) - height[i])
        return total
