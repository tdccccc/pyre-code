# Mistake: updates each row in place from left to right, so every entry adds an already-updated neighbour
class Solution:
    def generate(self, numRows):
        row, out = [1], []
        for _ in range(numRows):
            out.append(list(row))
            for j in range(1, len(row)):
                row[j] = row[j] + row[j - 1]
            row.append(1)
        return out
