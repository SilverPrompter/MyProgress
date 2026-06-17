import numpy as np

def softmax(x):
    e = np.exp(x-x.max(axis=-1,keepdims=True))
    return e / e.sum(axis=-1, keepdims= True)

def attention(Q,K,V):
    scores = Q @ K.T
    scores =  scores / np.sqrt(Q.shape[-1])
    weights = softmax(scores)
    return weights @ V

d_model = 4
d_ff = 16


W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

W1= np.random.randn(d_model,d_ff)
W2= np.random.randn(d_ff,d_model)


def feed_forward(x):
   expanded = x @ W1
   activated =np.maximum(0,expanded)
   shrunk = activated @ W2
   return shrunk

test_word = np.array([[1.0,0.0,0.5,0.0]])
print(feed_forward(test_word))
print(feed_forward(test_word).shape)

def layer_norm(x):
    mean= x.mean(axis=-1,keepdims=True)
    std = x.std(axis=-1, keepdims= True)
    return (x - mean)/ (std + 1e-9)

def encoder_block(x):
    attn_out = attention(x @ W_Q, x @ W_K, x @ W_V)
    x = x + attn_out
    x = layer_norm(x)

    ff_out = feed_forward(x)
    x = x + ff_out
    x = layer_norm(x)

    return x



# --- test it ---
X = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.5, 0.0],
    [0.0, 0.5, 1.0, 0.0],
])

out = encoder_block(X)
print(out)
print(out.shape)
