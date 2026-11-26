def dot(a,b):
    total = 0
    if len(a) != len(b):
        raise ValueError("Lengths are not equal " + str(len(a)) + " vs " + str(len(b)))
    for i in range(len(a)):
        total += a[i] * b [i]
    return total

def relu(z):
    if z > 0 :
        return z
    else:
        return 0

def neuron(x,w,b):
    dp =dot(x,w) + b
    return relu(dp)

import numpy as np

def neuron_np (x,w,b):
    x = np.array(x)
    w = np.array(w)
    dp = x @ w + b
    return relu(dp)


print(dot([3,2],[4,5]))
print(neuron([2, 3], [0.5, -1], 1))
print(neuron([1, 4], [2, -0.5], 0.5))
print(neuron([3, 1], [-1, 2], -0.5))
print(neuron_np([1, 4], [2, -0.5], 0.5))
print(neuron_np([2, 3], [0.5, -1], 1))      # should be 0
print(neuron_np([1, 4], [2, -0.5], 0.5))    # should be 0.5
print(neuron_np([3, 1], [-1, 2], -0.5))
