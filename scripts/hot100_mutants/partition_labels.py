# Mistake: cuts at each letter's own last occurrence, ignoring letters whose range extends the piece
class Solution:
    def partitionLabels(self, s):
        last = {c: i for i, c in enumerate(s)}
        sizes, start = [], 0
        for i, c in enumerate(s):
            if i == last[c]:
                sizes.append(i - start + 1)
                start = i + 1
        return sizes
