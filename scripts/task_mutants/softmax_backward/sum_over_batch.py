# Mistake: sums over the first (batch) dim instead of the softmax dim, mixing unrelated rows
def softmax_backward(y, dy):
    s = (dy * y).sum(dim=0, keepdim=True)  # BUG: should be dim=-1
    return y * (dy - s)
