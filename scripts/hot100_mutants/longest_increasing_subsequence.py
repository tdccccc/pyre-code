# Mistake: bisect_right lets an equal value extend the run, so repeats are counted
from bisect import bisect_right


class Solution:
    def lengthOfLIS(self, nums):
        tails = []
        for x in nums:
            k = bisect_right(tails, x)
            if k == len(tails):
                tails.append(x)
            else:
                tails[k] = x
        return len(tails)
