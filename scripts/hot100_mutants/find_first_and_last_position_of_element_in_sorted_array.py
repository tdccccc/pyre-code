# Mistake: returns the first match of a plain binary search as both ends of the range
class Solution:
    def searchRange(self, nums, target):
        lo, hi = 0, len(nums) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            if nums[mid] == target:
                return [mid, mid]
            if nums[mid] < target:
                lo = mid + 1
            else:
                hi = mid - 1
        return [-1, -1]
