# Mistake: the sum formula assumes the repeated value appears exactly twice; with k copies it gives dup * (k - 1)
class Solution:
    def findDuplicate(self, nums):
        n = len(nums) - 1
        return sum(nums) - n * (n + 1) // 2
