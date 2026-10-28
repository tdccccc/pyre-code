# Mistake: keeps a single running minimum, which is stale after the minimum is popped
class MinStack:
    def __init__(self):
        self.stack, self.low = [], None
    def push(self, val):
        self.stack.append(val)
        self.low = val if self.low is None else min(self.low, val)
    def pop(self):
        self.stack.pop()
        if not self.stack:
            self.low = None
    def top(self):
        return self.stack[-1]
    def getMin(self):
        return self.low
