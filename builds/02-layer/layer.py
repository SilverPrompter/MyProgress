def dot(a,b):
    total = 0
    if len(a) != len(b):
        raise ValueError("Lengths are not equal " + str(len(a)) + " vs " + str(len(b)))
    for i in range(len(a)):
        total += a[i] * b[i]
    return total

def relu(z):
    if z > 0:
        return z
    else:
        return 0

def column(w, j):
    nums = []
    for x in w:
        nums.append(x[j])
    return nums

def layer(x,W,b):
    outputs = []
    for j in range(len(W[0])):
        w_j = column(W,j)
        z = dot(x,w_j) + b[j]
        outputs.append(relu(z))
    return outputs


W = [[0.5, 2, 1], [-1, -0.5, 1]]
print(column(W, 0))   # should print [0.5, -1]
print(column(W, 2))   # should print [1, 1]
print(layer([2, 3], W, [1, 0.5, 0]))   # should print [0, 3.0, 5]
