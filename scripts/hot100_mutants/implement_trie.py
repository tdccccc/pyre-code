# Mistake: search checks the set of prefixes, so a prefix of a stored word counts as a stored word
class Trie:
    def __init__(self):
        self.prefixes = set()

    def insert(self, word):
        for i in range(1, len(word) + 1):
            self.prefixes.add(word[:i])

    def search(self, word):
        return word in self.prefixes

    def startsWith(self, prefix):
        return prefix in self.prefixes
