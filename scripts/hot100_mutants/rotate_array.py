# Mistake: slices with -k without reducing k modulo the length, so k > len(nums) leaves nums unrotated
class Solution:
    def rotate(self, nums, k):
        nums[:] = nums[-k:] + nums[:-k]
