# Mistake: sorts ascending and takes index k - 1, which is the k-th smallest rather than the k-th largest
class Solution:
    def findKthLargest(self, nums, k):
        return sorted(nums)[k - 1]
