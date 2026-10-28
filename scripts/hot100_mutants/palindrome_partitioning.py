# Mistake: the last piece never extends to the end of the string (range stops one short), so a palindrome suffix is never a piece
class Solution:
    def partition(self, s):
        result, pieces = [], []

        def backtrack(start):
            if start == len(s):
                result.append(pieces[:])
                return
            for end in range(start + 1, len(s)):
                piece = s[start:end]
                if piece == piece[::-1]:
                    pieces.append(piece)
                    backtrack(end)
                    pieces.pop()

        backtrack(0)
        return result
