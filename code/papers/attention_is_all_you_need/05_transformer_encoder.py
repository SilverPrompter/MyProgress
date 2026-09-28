import numpy as np

d_model = 4
d_ff = 16

def make_block_weights():
    return {
        "W_Q": np.random.randn(d_model, d_model),
        "W_K": np.random.randn(d_model, d_model),
        "W_V": np.random.randn(d_model, d_model),
        "W1" :np.random.randn(d_model,d_ff),
        "W2" :np.random.randn(d_ff,d_model)

    }

def softmax(x):
    e = np.exp(x-x.max(axis=-1, keepdims=True))
    return e/ e.sum(axis=-1, keepdims=True)

def attention(Q,K,V):
    scores =Q @K.T
    scores = scores /np.sqrt(Q.shape[-1])
    weights = softmax(scores)
    return weights @ V

def layer_norm(x):
    mean= x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1 , keepdims=True)
    return (x - mean) / (std + 1e-9)

def feed_forward(x, w):
    expanded = x @ w["W1"]
    activated = np.maximum(0, expanded)
    shrunk = activated @ w["W2"]
    return shrunk

def encoder_block(x, w):
    attn_out = attention(x @ w["W_Q"], x @ w["W_K"], x @ w["W_V"])
    x = x + attn_out
    x = layer_norm(x)

    ff_out = feed_forward(x, w)
    x = x + ff_out
    x = layer_norm(x)

    return x

num_blocks = 3
blocks = [make_block_weights() for _ in range(num_blocks)]

X  = np.array([
    [1.0,0.0,0.0,0.0],
    [0.0,1.0,0.5,0.0],
    [0.0,0.5,1.0,0.0],
    ])

x = X
for w in blocks:
    x = encoder_block(x,w)
print(x)
print(x.shape)

