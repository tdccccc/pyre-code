# Mistake: the deque never drops the front index once it has left the window, so an old maximum sticks
from collections import deque


class Solution:
    def maxSlidingWindow(self, nums, k):
        dq = deque()
        out = []
        for i, x in enumerate(nums):
            while dq and nums[dq[-1]] <= x:
                dq.pop()
            dq.append(i)
            if i >= k - 1:
                out.append(nums[dq[0]])
        return out
