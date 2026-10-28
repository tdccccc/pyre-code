# Mistake: get does not count as a use, so it never refreshes the key
from collections import OrderedDict
class LRUCache:
    def __init__(self, c):
        self.c, self.d = c, OrderedDict()
    def get(self, k):
        return self.d.get(k, -1)
    def put(self, k, v):
        if k in self.d:
            self.d.move_to_end(k)
        self.d[k] = v
        if len(self.d) > self.c:
            self.d.popitem(last=False)
