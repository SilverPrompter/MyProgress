import numpy as np

np.random.seed(0)

d_model = 4
d_ff = 16
tgt_len = 3   # decoder positions (English words being generated)
src_len = 4   # encoder positions (French words being read)

# ---------- from earlier sessions ----------
def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
    scores = Q @ K.T
    scores = scores / np.sqrt(Q.shape[-1])
    weights = softmax(scores)
    return weights @ V

def masked_attention(Q, K, V, mask):
    scores = Q @ K.T
    scores = scores / np.sqrt(Q.shape[-1])
    scores = np.where(mask == 1, -np.inf, scores)
    weights = softmax(scores)
    return weights @ V, weights

def layer_norm(x):
    mean = x.mean(axis=-1, keepdims=True)
    std = x.std(axis=-1, keepdims=True)
    return (x - mean) / (std + 1e-9)

def feed_forward(x):
    expanded = x @ W1
    activated = np.maximum(0, expanded)
    shrunk = activated @ W2
    return shrunk

# ---------- weights ----------
W_Q1 = np.random.randn(d_model, d_model)   # self-attention set
W_K1 = np.random.randn(d_model, d_model)
W_V1 = np.random.randn(d_model, d_model)

W_Q2 = np.random.randn(d_model, d_model)   # cross-attention set
W_K2 = np.random.randn(d_model, d_model)
W_V2 = np.random.randn(d_model, d_model)

W1 = np.random.randn(d_model, d_ff)        # feed-forward
W2 = np.random.randn(d_ff, d_model)

# ---------- the decoder block ----------
def decoder_block(x, enc_out, mask):
    # 1. masked self-attention — look at what I've generated so far
    self_out, _ = masked_attention(x @ W_Q1, x @ W_K1, x @ W_V1, mask)
    x = x + self_out
    x = layer_norm(x)

    # 2. cross-attention — look at the encoder's understanding of the source
    cross_out = attention(x @ W_Q2, enc_out @ W_K2, enc_out @ W_V2)
    x = x + cross_out
    x = layer_norm(x)

    # 3. feed-forward — each word processes alone
    ff_out = feed_forward(x)
    x = x + ff_out
    x = layer_norm(x)

    return x

# ---------- test ----------
mask = np.triu(np.ones((tgt_len, tgt_len)), k=1)

x = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.5, 0.0],
    [0.0, 0.5, 1.0, 0.0],
])

enc_out = np.random.randn(src_len, d_model)   # 4 words — a LONGER source

out = decoder_block(x, enc_out, mask)

print("decoder input shape: ", x.shape)
print("encoder output shape:", enc_out.shape)
print("decoder block output:")
print(out.round(3))
print("output shape:        ", out.shape)
