# Mistake: uses hi = len(nums) - 1 with lo < hi, so a target past the last element lands on the last index
class Solution:
    def searchInsert(self, nums, target):
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid
        return lo
