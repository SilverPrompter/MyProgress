# Day 07

## CS50P — Libraries

Watched the CS50P lecture on libraries — using code other people have already written instead of reinventing it.

- **import** — pull an existing library into your file so you can use its functions
- **from ... import ...** — pull in just the specific piece you need rather than the whole library
- **Standard library** — modules that ship with Python (random, statistics, sys, etc.) — no install needed
- **Third-party libraries** — installed with `pip` (like numpy, which I installed today for the attention work)
- **Making your own modules** — your own `.py` files can be imported into other files, which is how real projects stay organized

**Mental model:** a library is a toolbox someone already built and shared. `import` is reaching into the toolbox. No reason to forge your own hammer when a good one already exists.

This connected directly to today's ML work — I `pip install`ed numpy and `import`ed it to build the attention code.

---

## Multi-Head Attention — Built From Scratch (Session 2 of 7)

Implemented **multi-head attention** in pure numpy, building on Session 1's scaled dot-product attention. Every line written by hand.

### The concept
Session 1 ran attention once. Multi-head runs it several times in parallel — each run is a "head" with its own weight matrices. Each head can learn to notice a different kind of relationship between words. Afterward, all the heads' outputs are glued back together and mixed into one result.

### The key insight — splitting the dimensions
With 4 numbers per word and 2 heads, each head works with only 2 numbers per word:
- `d_model` (4) ÷ `num_heads` (2) = `d_k` (2) dimensions per head
- Each head runs its own attention in that smaller 2-D space
- Concatenate the heads back: 2 + 2 = 4, full size again

### The pipeline I built
| Step | What it does |
|------|-------------|
| `X` | Sentence — 3 words, 4 numbers each, shape (3, 4) |
| `W_Q, W_K, W_V` | A **list** of matrices — one set per head, each shaped (4, 2) |
| loop over heads | Each head makes its own Q/K/V and runs Session 1 attention |
| `np.concatenate(..., axis=1)` | Glue the head outputs side by side → back to (3, 4) |
| `W_O` | Final projection matrix (4, 4) that mixes the heads together |
| `combined @ W_O` | Final context-aware output, shape (3, 4) |

### The code
```python
import numpy as np

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def attention(Q, K, V):
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

W_O = np.random.randn(d_model, d_model)

head_outputs = []

for i in range(num_heads):
    Q = X @ W_Q[i]
    K = X @ W_K[i]
    V = X @ W_V[i]
    out = attention(Q, K, V)
    head_outputs.append(out)

combined = np.concatenate(head_outputs, axis=1)
output = combined @ W_O
print(output)
print(output.shape)
```

### Key lessons
- **List comprehension for weights** — `[np.random.randn(...) for _ in range(num_heads)]` builds one random weight matrix per head. Weights start random because in a real model they'd be *learned* through training; here they stay random since we're not training.
- **Fixed a hardcoding bug** — the scaling was `np.sqrt(4)` (correct for Session 1's size-4 vectors), but each head now works in a size-2 space. Changed it to `np.sqrt(Q.shape[-1])` so the function **reads its own size from the data** instead of assuming 4. Now it works at any dimension — better engineering habit.
- **Concatenate on axis=1** — joining along columns (the numbers-per-word direction) is what glues the heads back to full width.

This is the same mechanism running inside every GPT and Claude model — just with more heads and far bigger numbers.

**Next session:** positional encoding (Session 3 of 7).

---

## Note
- Leet Coding was done today by me and I did a problem where I was rotating a matrix 90 degree s and checking if it could match a target matrix. I learned to transpose with zip(*grid) then reverse [::-1] to achive the result of rotating the grid 90 degrees it also was a little lesson in chaning functions as zip(*grid) was returning it as a tuple so i had to convert it back into a list 

---

## Patterns and Connections Today
- CS50P libraries → directly used today: `pip install numpy` + `import numpy` powered the attention build
- Multi-head attention → extends Session 1; same dot-product-as-similarity core, now run in parallel across heads
- Reading size from data (`Q.shape[-1]`) instead of hardcoding → a defensive-coding habit that carries over from security work: don't assume, measure

---

## Repo Updates Needed
- [ ] Add this log to `/logs` folder as `day_07.md`
- [ ] Update README Recent Activity (newest on top)
- [ ] Add `02_multihead_attention.py` to repo (Session 2 of transformer build)
