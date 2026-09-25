# Day 11

## Transformer From Scratch — Session 6 (Part 1): The Decoder & Masked Attention

Started the decoder — the generation half of the Transformer. Built and proved causal masking. Most of the session was conceptual: working out *why* the architecture is shaped the way it is.

---

## Concept Review — Reviving Intuition

Came back after a break and rebuilt the mental model by tracing the encoder as a **chain of problems**, where each component exists to fix something the previous one broke:

```
words must become numbers            → embeddings
but embeddings are static            → attention (mix words together)
but what decides the mixing?         → dot product = similarity scores
but raw scores can be negative       → softmax (positive, sums to 1)
but one vector can't do 3 jobs       → Q / K / V split
but one pass sees one pattern        → multi-head attention
but attention is order-blind         → positional encoding
but words never think alone          → feed-forward network
but stacking overwrites / kills grads → residual connections
but values drift in scale            → layer normalization
and now shapes match                 → stack the blocks
```

Nothing in the architecture is decorative. Pull one piece out and something breaks.

---

## Deep Dive 1 — Why Q/K/V Must Be Separate

**The failure without it:** "The cat sat on the mat. **It** was tired." To resolve "it," the model must pick *cat* over *mat*. With a single vector per word, the only question askable is "whose embedding is most similar to 'it'?" — and *cat* and *mat* are both short concrete nouns with nearby embeddings. Similarity alone can't separate them.

**With three projections**, the question changes shape:
- "It" uses its **Query** to ask: *I'm a pronoun, I need an animate antecedent that could be tired*
- "Cat" uses its **Key** to advertise: *animate, subject, can be tired*
- "Mat" uses its **Key** to advertise: *inanimate, location, surface*

The match becomes *what I'm looking for* vs *what you're offering* — two separately-learned spaces, not one embedding compared against itself.

**The sharpest framing:**
> **Q and K control where attention goes. V controls what gets delivered when it arrives.**

Merge them and the thing that makes a word *attractive to attend to* is forced to be identical to the thing it *contributes*. Those are genuinely different jobs.

```sql
SELECT  content   FROM words  WHERE  tag  MATCHES  search
          ↑                            ↑              ↑
        VALUE                         KEY           QUERY
```

---

## Deep Dive 2 — Why the Feed-Forward Network Exists

Every operation in attention is a **weighted sum of other words' vectors**. That means attention can only ever produce *combinations of things already present*. It cannot compute anything genuinely new about a single word in isolation.

**The concrete gap:** after attention, "bank" has absorbed river-signal from its neighbors. But absorbing ≠ concluding. Something still has to perform:

> *I default to "financial institution," I'm now carrying strong river-signal → therefore become "riverbank"*

That's a transformation on one word's own contents. Attention structurally can't do it — its only move is averaging.

```
attention:     words talk to each other      (BETWEEN words)
feed-forward:  each word processes alone     (WITHIN a word)
```

The two alternate layer after layer: **gather, digest, gather, digest.**

**Why expand 4→16→4?** A plain matrix multiply only draws straight lines. The ReLU is the one non-linear "bend" — but a bend needs *room to bend in*. In cramped 4-D the data points pile on top of each other; lifted to 16-D they spread out and the bend can cleanly separate them. Then shrink back, carrying the result.

*Knot analogy:* tangled string flat on a table looks like a merged mess from above. Lift it into 3D and the strands visibly separate. Set it back down. You didn't keep the dimension — but working up there revealed structure the flat view couldn't show.

---

## Deep Dive 3 — Cross-Attention and the Encoder/Decoder Relationship

**Cross-attention:** Q comes from the decoder, K and V come from the **encoder output**. The decoder asks; the encoder's understanding answers.

```
self-attention:   attention(decoder, decoder, decoder)
cross-attention:  attention(decoder, encoder, encoder)
                       ↑         ↑
                    Q only    K and V
```

K and V always travel together — one is the index, one is the content. You'd never index sentence A but return content from sentence B.

### Big question worked out: does the encoder "know English"?

**No.** Nobody hands it a dictionary. The encoder outputs vectors that are neither French nor English — a **language-neutral representation of meaning**. The French word "chat" becomes a vector for the *concept*, not the word.

The alignment is **learned, not given**. Encoder and decoder are trained **jointly, end to end**, on the same loss. Every wrong English word pushes weights on *both* sides. Over millions of pairs, they co-evolve a shared private code — because the loss punished them whenever they failed to communicate.

> They didn't learn each other's languages. They invented a shorthand under pressure.

### Why not skip the decoder and translate directly?

- **Word counts don't match** — "pomme de terre" (3 words) → "potato" (1). The decoder generates freely; the encoder outputs exactly one vector per input word.
- **Word order differs** — "le chat noir" → "the black cat." Direct positional mapping produces garbage.
- **Output depends on prior output** — "the cats" forces "are," not "is." Only the decoder sees what's been generated so far.
- **Meaning is distributed** — one English word may depend on five scattered French words. Cross-attention gathers from wherever, in any combination.

---

## Deep Dive 4 — When Does the "Note-Checking" Happen?

Two separate things at two separate times:

**Weights get shaped — ONCE, during training.** Start random, adjusted by backprop, then frozen forever. No learning happens at runtime.

**Cross-attention lookups — EVERY generated word, at runtime.**

```
ONCE per sentence:   encoder reads the French → vectors sit frozen in memory

ONCE per word,       decoder consults those vectors via cross-attention
per layer:           (asking a DIFFERENT question each time)
```

Generating "cat" the Query asks *what's the subject noun?* Generating "sat" it asks *what's the verb?* — same frozen notes, different question, different parts light up.

With a 6-layer decoder producing 10 words: **60 cross-attention lookups** against one encoder pass.

```
weights = the learned skill to ask good questions   (permanent)
vectors = this specific sentence's content          (fresh every time)
```

---

## Deep Dive 5 — Why Split Encoder/Decoder At All?

Pushed on why two systems instead of one. **Correct instinct — the field agreed and moved to decoder-only.**

**Why the split made sense in 2017:** the two halves have *opposite* constraints. The encoder should be fully bidirectional (word 1 sees word 10, word 10 sees word 1) — that's the right way to understand text. The decoder must be strictly causal, never seeing the future. Separating them let each be optimal for its own job. Plus efficiency: the encoder runs once per sentence, the decoder runs once per word.

**Why decoder-only won:** concatenate everything into one stream and run causal attention over all of it —

```
[French sentence] [separator] [English translation]
```

The French isn't "notes" anymore — it's just earlier context. Same attention, same weights, no cross-attention, no separate stack, no lossy handoff. One uniform mechanism, one objective (predict next token), trainable on *any* text rather than paired data. That's what made GPT-3 and everything after possible.

Encoder-decoder still wins where input and output are genuinely separate objects — T5, Whisper (speech-to-text), translation, summarization.

---

## The Code — Causal Masking

**The idea:** block everything in the "future" direction by setting those scores to `-inf` **before** softmax. Then `exp(-inf) = 0`, so they get exactly zero weight.

```python
import numpy as np

seq_len = 3
d_model = 4

def softmax(x):
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)

def masked_attention(Q, K, V, mask):
    scores = Q @ K.T
    scores = scores / np.sqrt(Q.shape[-1])
    scores = np.where(mask == 1, -np.inf, scores)
    weights = softmax(scores)
    return weights @ V, weights

mask = np.triu(np.ones((seq_len, seq_len)), k=1)

X = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.5, 0.0],
    [0.0, 0.5, 1.0, 0.0],
])

W_Q = np.random.randn(d_model, d_model)
W_K = np.random.randn(d_model, d_model)
W_V = np.random.randn(d_model, d_model)

Q = X @ W_Q
K = X @ W_K
V = X @ W_V

out, weights = masked_attention(Q, K, V, mask)

print(weights)
print(out.shape)
```

**`np.triu(np.ones((seq_len, seq_len)), k=1)`** builds the blocking map — "triangle upper," starting one above the diagonal:

```
[[0. 1. 1.]      The blocks cat, sat
 [0. 0. 1.]      cat blocks sat
 [0. 0. 0.]]     sat blocks nothing
```

### Proof it works

```
        The    cat    sat
The  [ 1.00   0.00   0.00 ]   only itself — nothing else exists yet
cat  [ 0.55   0.45   0.00 ]   sees The + itself; "sat" invisible
sat  [ 0.38   0.15   0.48 ]   sees everything before it
```

Three things this proves:
- **Upper triangle is exactly zero** — not small, zero. The `-inf` trick worked.
- **Row 0 is `[1, 0, 0]`** — the first word has only itself; softmax gives 100% self-attention by necessity.
- **Every row still sums to 1** — softmax redistributed full weight among only the *allowed* positions. Nothing leaked, nothing lost.

### Critical ordering

```
mask FIRST   → future scores become -inf
softmax NEXT → exp(-inf) = 0 → zero weight
```

Reversed, the weights would no longer sum to 1 — you'd be zeroing values that had already been normalized.

This triangular pattern is the single most important structure in modern language models. It's the same mask running inside GPT and Claude right now, just at scale.

---

## Patterns and Connections Today

- Q/K/V separation ↔ database SELECT / WHERE / index — search term, index, and stored rows are deliberately three different things
- FFN expand-then-shrink ↔ lifting a knot into 3D to untangle it
- Masked attention ↔ the foundation of every decoder-only model (GPT, Claude) — cross-attention is the piece that got dropped, and understanding it is what makes *why* it got dropped visible
- Reasoning from first principles about architecture tradeoffs landed on the field's real historical conclusion

---

## Still To Build (Session 6, Part 2)

- [ ] Cross-attention (same attention function, K/V sourced from the encoder)
- [ ] Full decoder block: masked self-attn → cross-attn → FFN, each with residual + layer_norm
- [ ] Session 7: tiny training run, forward + backward end to end

---

## Repo Updates Needed

- [ ] Add this log to `/logs` folder as `day_11.md`
- [ ] Update README Recent Activity (newest on top)
- [ ] Add `06_decoder.py` to repo (Session 6, in progress)
