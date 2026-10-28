# Mistake: never checks that ")" does not outnumber "(", so strings like ")(" slip through
class Solution:
    def generateParenthesis(self, n):
        result = []

        def backtrack(current, opened, closed):
            if len(current) == 2 * n:
                result.append(current)
                return
            if opened < n:
                backtrack(current + "(", opened + 1, closed)
            if closed < n:
                backtrack(current + ")", opened, closed + 1)

        backtrack("", 0, 0)
        return result
