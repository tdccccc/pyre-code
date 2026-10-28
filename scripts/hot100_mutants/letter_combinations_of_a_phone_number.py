# Mistake: empty digits give [""] instead of [], because the partial results start as [""] and are never special-cased
class Solution:
    def letterCombinations(self, digits):
        keys = {"2": "abc", "3": "def", "4": "ghi", "5": "jkl",
                "6": "mno", "7": "pqrs", "8": "tuv", "9": "wxyz"}
        combos = [""]
        for d in digits:
            combos = [c + ch for c in combos for ch in keys[d]]
        return combos
