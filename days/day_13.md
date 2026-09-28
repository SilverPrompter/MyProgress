# Day 13

**Milestone: finished implementing *Attention Is All You Need* from scratch. All 7 sessions.**

---

## LeetCode

### Index-expression drill — 5/6

Opened with a deliberate drill instead of a new problem, targeting the recurring bug from the last two sessions: confusing arithmetic *inside* brackets with arithmetic *outside* them.

**The rule:**
> Inside the brackets is a **position**. Outside the brackets is **arithmetic on the value**.
> `nums[x+1]` moves where you're looking. `nums[x] + 1` changes what you found.

5/6 correct on expressions that had previously caused repeated failures. The bug is closing out.

### Destroying Asteroids — greedy with sorting

Pattern: sort ascending, then consume in order, because each success expands what's reachable next.

Also covered:
- **`sorted()` vs `.sort()`** — `sorted()` returns a new list; `.sort()` mutates in place and returns `None`. Assigning the result of `.sort()` gives you `None` — a classic trap.
- **Verifying greedy with brute force** — ran 3000 random trials against a brute force over all orderings: 0 mismatches. Reusable technique: any time a greedy strategy is *claimed* optimal, brute-force the full search space on tiny inputs. Turns "I think this works" into evidence. This is differential testing.

---

## Transformer From Scratch — Session 6: The Decoder

### The setup: a different job

The encoder **understands** — it reads everything at once, bidirectionally. The decoder **generates** — one token at a time, left to right. Opposite constraints, which is why they're built differently.

### Masked (causal) attention

**The problem:** during training you feed the decoder the whole target sentence at once. Without a mask, position 4 would attend to position 5 and learn to "predict" it by copying the answer. Then at real generation time those future tokens don't exist and the model collapses. That's cheating — studying with the answer key, then sitting a blank exam.

**The fix:** set future scores to `-inf` **before** softmax. Since `exp(-inf) = 0`, those positions get exactly zero weight.

```python
def masked_attention(Q, K, V, mask):
    scores = Q @ K.T
    scores = scores / np.sqrt(Q.shape[-1])
    scores = np.where(mask == 1, -np.inf, scores)
    weights = softmax(scores)
    return weights @ V, weights

mask = np.triu(np.ones((seq_len, seq_len)), k=1)
```

`np.triu(..., k=1)` builds the blocking map — "triangle upper," starting one above the diagonal.

**Proof it works:**
```
        The    cat    sat
The  [ 1.00   0.00   0.00 ]   only itself — nothing else exists yet
cat  [ 0.55   0.45   0.00 ]   sees The + itself; "sat" invisible
sat  [ 0.38   0.15   0.48 ]   sees everything before it
```

Upper triangle is exactly zero — not small, zero. And **every row still sums to 1**: softmax redistributed full weight among only the allowed positions. Nothing leaked.

**Critical ordering:** mask FIRST, softmax SECOND. Reversed, the weights would no longer sum to 1 — you'd be zeroing values that had already been normalized.

### Cross-attention

Q from the decoder, K and V from the encoder output. The decoder asks; the encoder's understanding answers.

```python
cross_out = attention(x @ W_Q2, enc_out @ W_K2, enc_out @ W_V2)
```

**Got the mask question wrong first, and the correction was instructive.** My instinct was that cross-attention *should* be masked, reasoning that differing word order might cause confusion. The opposite is true, and my own reasoning proved it:

```
French:   le(1)   chat(2)   noir(3)
English:  the(1)  black(2)  cat(3)
```

Generating "black" at English position 2 requires "noir" at French position **3**. A causal mask would block it. The mask wouldn't prevent word-order confusion — it would **guarantee failure** whenever the order differs, which is most of the time.

**The rule:**
> Mask the thing you're **generating**. Never mask the thing you're **reading**.

### The full decoder block — three sublayers

```python
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
```

Every sublayer ends with the same two moves: **add its own output back, then normalize.** Three times per block. Once you see the rhythm it's mechanical.

*(Bug caught: had `x = x + self_out` inside sublayer 2, which would throw cross-attention's result away and add self-attention's twice. Each sublayer adds back **its own** output.)*

### The payoff — mismatched lengths

Deliberately tested with 4 encoder positions against 3 decoder positions:

```
decoder input shape:  (3, 4)
encoder output shape: (4, 4)
output shape:         (3, 4)
```

Cross-attention doesn't care that source and target lengths differ. **The Q side sets the output length; the K/V side can be any length at all.** That's the "pomme de terre → potato" problem, solved in code — and another reason a causal mask there would be wrong.

---

## Transformer From Scratch — Session 7: Training

Everything before this was the **forward** pass — data flowing through. Training adds the missing half: actually changing the weights.

### Three ideas

**1. Loss — one number for "how wrong are we."**

```python
def cross_entropy(probs, targets):
    rows = np.arange(len(targets))       # [0, 1, 2] — the row numbers
    correct = probs[rows, targets]       # pair (row, target) -> grab that value
    loss = -np.mean(np.log(correct + 1e-9))
    return loss
```

Pull out the probability assigned to the **correct** token, take the log, negate. Confident and right → loss near zero. Confident and wrong → loss large. "How surprised was the model by the truth?"

Verified against a good case and a bad case:
```
targets [0, 2, 1]  -> picks [0.7, 0.7, 0.6]  -> loss 0.4081   (right)
targets [1, 0, 3]  -> picks [0.1, 0.1, 0.1]  -> loss 2.3026   (wrong)

being wrong costs 5.6x more loss
```

**Why log?** It stretches the low end so catastrophic wrongness hurts disproportionately:
```
probability 0.9    -> loss 0.105
probability 0.5    -> loss 0.693
probability 0.1    -> loss 2.303
probability 0.01   -> loss 4.605
probability 0.001  -> loss 6.908
```
Not linear. 0.9 → 0.5 costs ~0.6. But 0.1 → 0.001 costs 4.6.

**2. Gradient — nudge a weight and see what happens.**

```python
flat[i] = original + eps
loss_up = total_loss(p)

flat[i] = original - eps
loss_down = total_loss(p)

gflat[i] = (loss_up - loss_down) / (2 * eps)
```

That *is* a gradient — bump the weight up, measure; bump it down, measure; the difference says which direction helps and how strongly. No calculus, just measurement.

**3. Update — step downhill.**

```python
params[name] = params[name] - learning_rate * grads[name]
```

Subtract the gradient. If increasing a weight made things worse, the gradient is positive, so subtracting moves it down.

### It learns

Task: predict the previous token. Solvable only by attending backwards — attention is *required*.

```
starting loss: 1.4546
step 0    1.3952
step 45   0.0402
step 150  0.0012

input [3, 1, 2] -> predicted [3, 3, 1]   target [3, 3, 1]   OK
input [0, 2, 1] -> predicted [0, 0, 2]   target [0, 0, 2]   OK
input [1, 3, 0] -> predicted [1, 1, 3]   target [1, 1, 3]   OK
input [2, 0, 3] -> predicted [2, 2, 0]   target [2, 2, 0]   OK

4/4 sequences exactly right
```

Random matrices became **meaningful** matrices. That's training.

### Why backpropagation exists

This model has 80 weights. Two loss measurements per weight = **160 forward passes for a single step.**

GPT-3 has 175 billion weights. One step this way = 350 billion forward passes. At a millisecond each, roughly **11,000 years per step.**

Backpropagation gets every gradient in one backward pass — about the cost of one forward pass, for all weights simultaneously. That isn't an optimization. That's the difference between deep learning existing and not existing.

---

## Python / numpy lessons that actually cost me time today

- **`(` vs `[`** — `np.array(...)` is a *function call*, parentheses. `probs[...]` is *indexing*, brackets. Got this backwards repeatedly. And `np.array([...])` needs **both**: parens to call, brackets to bundle the rows into one argument.
- **Parameters vs hardcoded values** — a function with its data defined inside can only ever do one thing. `def cross_entropy(probs, targets)` takes them as inputs, so the same function serves the good case and the bad case.
- **Assignment shape** — `new_name = function(input)`. Left of `=` is what you're creating; it never goes inside the parens.
- **Indentation is syntax, not style** — every line in a function body must sit at the same column. A `return` at a shallower indent is *outside* the function and errors.
- **Targets are indices, not values** — a target says *which slot* holds the right answer (0, 1, 2, 3), not how confident the model was (0.7, 0.1). There is no "token 0.7."
- **Vectorization — no loop needed.** `probs[rows, targets]` pairs two lists element-wise inside numpy's compiled code. I'd been relying on this since session 1 without noticing: every `@` performs hundreds of multiply-adds with no visible loop. Rule of thumb: if you're writing a `for` loop over data, there's usually a vectorized way that's shorter and faster. This is also *why* the Transformer beat RNNs — attention is expressible as big matrix multiplies that parallelize; RNNs force sequential steps.

---

## Conceptual deep dives (carried over from the decoder work)

**Why Q/K/V must be separate.** "The cat sat on the mat. **It** was tired." With one vector per word, the only askable question is "whose embedding is most similar to 'it'?" — and *cat* and *mat* are both short concrete nouns sitting close together. Three projections change the question's shape: "it" **queries** for an animate antecedent, "cat" **advertises** animate/subject, "mat" advertises inanimate/location.

> **Q and K control where attention goes. V controls what gets delivered when it arrives.**

**Why the feed-forward exists.** Every operation in attention is a weighted sum of *other words' vectors* — so attention can only produce combinations of things already present. Something still has to perform *"I default to financial-institution, I'm carrying river-signal, therefore become riverbank."* That's a transformation on one word's own contents, and attention structurally cannot do it.

```
attention:     words talk to each other      (BETWEEN words)
feed-forward:  each word processes alone     (WITHIN a word)
```
The two alternate: **gather, digest, gather, digest.**

**Does the encoder "know English"?** No. Nobody hands it a dictionary. It outputs vectors that are neither French nor English — a language-neutral representation of *meaning*. Encoder and decoder are trained **jointly, end to end**, on one loss. They co-evolved a shared private code because the loss punished them whenever they failed to communicate. *They didn't learn each other's languages — they invented a shorthand under pressure.*

**When does the "note-checking" happen?** Two separate things:
```
ONCE, during training:   weights get shaped, then frozen forever
ONCE per sentence:       encoder reads source -> vectors sit frozen in memory
ONCE per word, per layer: decoder consults those vectors via cross-attention
```
Generating "cat" the Query asks *what's the subject noun?*; generating "sat" it asks *what's the verb?* — same frozen notes, different question. A 6-layer decoder producing 10 words does **60 cross-attention lookups** against one encoder pass.

**Why split encoder/decoder at all?** Pushed on this and reasoned to the field's actual answer: **decoder-only won.** Concatenate everything into one stream, run causal attention over all of it — the source isn't "notes" anymore, it's just earlier context. No cross-attention, no separate stack, no lossy handoff. One uniform mechanism, one objective, trainable on *any* text rather than paired data. That's what made GPT-3 possible. Encoder-decoder still wins where input and output are genuinely separate objects (T5, Whisper).

---

## Paper complete — all 7 sessions

| Session | File | Built |
|---------|------|-------|
| 1 | `01_attention.py` | scaled dot-product attention |
| 2 | `02_multihead_attention.py` | parallel heads, concat, output projection |
| 3 | `03_positional_encoding.py` | sin/cos position fingerprints |
| 4 | `04_encoder_block.py` | attention + FFN + residual + layer_norm |
| 5 | `05_transformer_encoder.py` | stacked blocks, per-block weights |
| 6 | `06_decoder.py` | masked self-attn + cross-attn + FFN |
| 7 | `07_loss.py`, `07_train_tiny.py` | cross-entropy, gradients, it learns |

Both halves of the architecture, plus a working training loop, in pure numpy. Roughly 60 lines of actual mechanism.

---

## Connections

- Greedy-with-sorting ↔ chaining footholds in an engagement: take the cheap wins first, each one expands what's reachable next
- Brute-force cross-checking ↔ differential testing: run two implementations against each other and look for disagreement
- Masked attention ↔ the foundation of every decoder-only model — cross-attention is the piece that got dropped, and understanding it is what makes *why* visible
- Vectorization ↔ the Transformer's core advantage over RNNs: parallelizable matrix multiplies vs forced sequential steps
- Index-vs-value drill ↔ the same precision the ML code demands (`W_Q1` vs `W_Q2`, `self_out` vs `cross_out` — both caught today)

---

## Next

- Paper 2 of 7: **GPT-3, "Language Models are Few-Shot Learners"** (Brown et al., 2020) — what happens when you scale this architecture ~1000x
- Optionally revisit session 7 with real backpropagation instead of numerical gradients

---

