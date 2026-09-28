import numpy as np

def softmax(x):
    e = np.exp(x -x.max(axis=-1, keepdims=True))
    return e /e.sum(axis=-1 ,keepdims=True)

def attention(Q,K,V):
    scores = Q @ K.T
    scores = scores / np.sqrt(Q.shape[-1])
    weights = softmax(scores)
    output = weights @ V
    return output

X = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.5, 0.0],
    [0.0, 0.5, 1.0, 0.0],
])


d_model = 4
num_heads = 2
d_k = d_model // num_heads

W_Q = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
W_K = [np.random.randn(d_model, d_k) for _ in range(num_heads)]
W_V = [np.random.randn(d_model, d_k) for _ in range(num_heads)]

head_outputs= []

for i in range(num_heads):
    Q = X @ W_Q[i]
    K = X @ W_K[i]
    V = X @ W_V[i]
    out = attention(Q, K, V)
    head_outputs.append(out)

combined = np.concatenate(head_outputs, axis = 1)
print(combined.shape)


W_O = np.random.randn(d_model, d_model)

output = combined @ W_O
print(output)
print(output.shape)
