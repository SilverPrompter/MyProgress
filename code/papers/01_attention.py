import numpy as np

#concept building version

X = np.array([
    [1.0,0.0,0.0,0.0],
    [0.0,1.0,0.5,0.0],
    [0.0,0.5,1.0,0.0],
])

W_Q =np.random.rand(4,4,)
W_K = np.random.rand(4,4)
W_V = np.random.rand(4,4)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

# scores = Q @ K.T
#
# print(scores.shape)
#
# scores = scores /np.sqrt(4)
#
def softmax(x):
     e = np.exp(x-x.max(axis=1 , keepdims=True))
     return e /e.sum(axis=1 ,keepdims=True)
#
# weights = softmax(scores)
# print(weights)
#
# output = weights @ V
# print(output)
# print(output.shape)

#function version

def attention(Q,K,V):
    scores = Q @ K.T
    scores = scores / np.sqrt(4)
    weights = softmax(scores)
    output = weights @ V
    return output


result = attention(Q,K,V)
print(result)
