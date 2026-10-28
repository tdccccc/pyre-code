# Mistake: alternates pushes between the two heaps without rebalancing, so the top of the larger heap is not the median
import heapq


class MedianFinder:
    def __init__(self):
        self.small, self.large = [], []  # small: max-heap via negation

    def addNum(self, num):
        if len(self.small) <= len(self.large):
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)

    def findMedian(self):
        if len(self.small) > len(self.large):
            return float(-self.small[0])
        return (-self.small[0] + self.large[0]) / 2
