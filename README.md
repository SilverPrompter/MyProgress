# Zero to ML Engineer — A Learning Journey

> From penetration tester to AI/ML engineer. Goal: Anthropic. Timeline: 17 weeks.

---

## Progress Overview

| Phase | Focus | Progress |
|-------|-------|----------|
| Foundations | Python + CS vocabulary | 🟡 `██████░░░░` 65% |
| 0 | Python stdlib + KV diagnostic | ⬜ `░░░░░░░░░░` 0% |
| 1 | System building drills | ⬜ `░░░░░░░░░░` 0% |
| 2 | Public eval harness + blog | ⬜ `░░░░░░░░░░` 0% |
| 3 | System design + Kleppmann | ⬜ `░░░░░░░░░░` 0% |
| 4 | AI safety + STAR method | ⬜ `░░░░░░░░░░` 0% |
| 5 | Mock interviews + apply | ⬜ `░░░░░░░░░░` 0% |

**Parallel track:** ML systems and internals study throughout all phases.

---

## What I'm Using

**Courses**
- Harvard CS50P — Python + CS fundamentals
- Khan Academy — Mathematics (working up to linear algebra and calculus)

**Video**
- 3Blue1Brown — Neural networks, linear algebra, calculus series
- Computerphile — CS concepts

**Books**
| Title | Author | Status |
|-------|--------|--------|
| AI Engineering | Chip Huyen | ✅ Finished |
| Build a Large Language Model From Scratch | Sebastian Raschka | 🟡 In progress |
| LLM Engineer's Handbook | — | 🟡 In progress |
| 50 Algorithms Every Programmer Should Know | — | 🟡 In progress |
| Prompt Engineering for Generative AI | — | 🟡 In progress |
| Designing Machine Learning Systems | — | 🟡 In progress |
| Designing Data-Intensive Applications | Martin Kleppmann | 🟡 In progress |

**Practice**
- Daily LeetCode — learning-first workflow, one problem per day
- Papers implemented from scratch — full architectures in NumPy, every line written by hand

---

## Research Papers

| # | Paper | Authors | Read | Coded |
|---|-------|---------|------|-------|
| 1 | DataPerf: Benchmarks for Data-Centric AI | Mazumder et al. | ✅ | ⬜ |
| 2 | GPT-3: Language Models are Few-Shot Learners | Brown et al. | ✅ | ⬜ |
| 3 | Attention Is All You Need | Vaswani et al. | ✅ | ✅ |
| 4 | AdderNet: Do We Really Need Multiplications? | Chen et al. | ✅ | ⬜ |
| 5 | Small Language Models are the Future of Agentic AI | Belcak et al. | ✅ | ⬜ |

---

## Transformer From Scratch ✅

*Attention Is All You Need (Vaswani et al., 2017) — implemented end to end in pure NumPy. Both halves of the architecture plus a working training loop.*

| Session | File | Built |
|---------|------|-------|
| 1 | [`01_attention.py`](transformer/01_attention.py) | Scaled dot-product attention — Q/K/V, scores, softmax, weighted blend |
| 2 | [`02_multihead_attention.py`](transformer/02_multihead_attention.py) | Parallel heads, concatenation, output projection |
| 3 | [`03_positional_encoding.py`](transformer/03_positional_encoding.py) | Sin/cos position fingerprints |
| 4 | [`04_encoder_block.py`](transformer/04_encoder_block.py) | Attention + FFN + residual connections + layer norm |
| 5 | [`05_transformer_encoder.py`](transformer/05_transformer_encoder.py) | Stacked blocks with independent per-block weights |
| 6 | [`06_decoder.py`](transformer/06_decoder.py) | Masked self-attention + cross-attention + FFN |
| 7 | [`07_loss.py`](transformer/07_loss.py) · [`07_train_tiny.py`](transformer/07_train_tiny.py) | Cross-entropy loss, gradients, a model that learns |

**Results:** the tiny model trains from loss 1.45 → 0.001 and solves its task 4/4. Proved attention is order-blind (shuffled sentences gave byte-identical output), then proved positional encoding fixes it. Proved causal masking zeroes the upper triangle while rows still sum to 1.

---

## Certifications

| Certificate | Provider | Status |
|-------------|----------|--------|
| AWS Certified Machine Learning — Associate | Amazon | 🟡 In progress |
| AWS Certified Machine Learning — Specialty | Amazon | ⬜ Planned (~1 year) |

---

## Daily LeetCode Log

| Day | Problem | Pattern | Difficulty |
|-----|---------|---------|------------|
| 01 | Robot Return to Origin | Running counter balance | Easy |
| 02 | Check if Strings Can Be Made Equal | Group and compare | Easy |
| 03 | Matrix Cyclic Shift | Cyclic transformation | Medium |
| 04 | Grid Equal Sum Partition | Prefix sum | Medium |
| 05 | Flip Submatrix | Two pointer reversal | Medium |
| 06 | Biggest Rhombus Sums | Simulation + deduplication | Medium |
| 07 | Find Missing Binary String | Cantor's diagonal argument | Medium |
| 08 | Check if Array is Sorted and Rotated | Rotation point detection | Easy |
| 09 | Partition Array According to Given Pivot | Three-way partition | Medium |
| 10 | Matrix Rotation | Transpose + row reversal | Medium |
| 11 | Largest Submatrix with Rearrangements | Histogram heights + greedy sort | Medium |
| 12 | Number Complement | Bitmask XOR | Easy |
| 13 | Path Existence Queries in a Graph | Connected components via labeling | Medium |
| 14 | Destroying Asteroids | Greedy with sorting | Medium |

**Parked:** Maximum Total Value of K Distinct Subarrays (Hard) — explored at the intuition level, best-first search with a heap. Returning to it once heap mechanics are solid.

---

## Patterns Learned

- **Running counter balance** — initialize, modify in loop, check final state
- **Group and compare** — split into independent groups, check equivalence
- **Cyclic transformation** — use `k % n` for effective shift, slice to rotate
- **Prefix sum** — running total checked against a target at each step
- **Two pointer reversal** — one pointer each end, moving inward
- **Simulation + deduplication** — simulate every case, set handles duplicates
- **Cantor's diagonal argument** — construct a value guaranteed to differ from every item
- **Rotation point detection** — count drops, check the wrap
- **Three-way partition** — split into less, equal, greater buckets preserving order
- **Transpose + reverse** — rotate a matrix with `zip(*mat)` then reverse each row
- **Histogram heights + greedy sort** — build per-cell heights, sort descending per row, maximize `height × width`
- **Bitmask XOR** — flip all bits within a width by XOR against an all-ones mask
- **Connected components via labeling** — label reachable groups, then answer queries by comparing labels
- **Greedy with sorting** — sort ascending and consume in order, because each success expands what's reachable next

**Verification technique:** when a greedy strategy is *claimed* optimal, brute-force the full search space on tiny inputs. 3000 random trials vs. all orderings → 0 mismatches. Turns "I think this works" into evidence. This is differential testing.

---

## ML Systems & Internals Progress

**Transformer architecture**
- [x] Token and positional embeddings
- [x] Layer normalization
- [x] GELU activation
- [x] Multi-head attention — Query, Key, Value mechanics
- [x] Feedforward networks
- [x] Residual connections
- [x] Output head and text generation loop
- [x] Parameter counting
- [x] Causal masking and cross-attention
- [x] Loss functions and cross entropy
- [x] Training loop — forward, loss, gradient, update
- [ ] Backpropagation — gradients built numerically so far; real backprop still to come

**Deep learning foundations**
- [x] McCulloch–Pitts neuron (1943) and Rosenblatt's perceptron (1958)
- [x] Activation functions and why nonlinearity is required
- [x] CNN pipeline — convolution, max pooling, flatten, dropout, softmax

---

## Recent Activity

*Newest first — full session details in the [/logs](logs) folder.*

- **Day 13** — index-expression drill (5/6, the recurring bug is closing out), Destroying Asteroids via greedy-with-sorting, **completed *Attention Is All You Need* from scratch** — decoder with masked + cross-attention, cross-entropy loss, and a training run that actually learns (7/7 sessions)
- **Day 12** — deep learning foundations: McCulloch–Pitts neuron (1943), Rosenblatt's perceptron (1958), activation functions, full CNN pipeline; spaced-retrieval re-attempt of Check if Array Is Sorted and Rotated (pattern not retained — logged honestly)
- **Day 11** — Transformer session 6: causal masking built and proven; deep dives on Q/K/V separation, the feed-forward's purpose, cross-attention, and why decoder-only architectures won
- **Day 10** — _(fill in from `logs/day_10.md`)_
- **Day 09** — Transformer session 5: stacked encoder blocks into a full encoder with independent per-block weights
- **Day 08** — Transformer sessions 3 & 4: positional encoding (sin/cos "clock dials"), full encoder block with residuals and layer norm
- **Day 07** — CS50P libraries; Transformer session 2: multi-head attention
- **Day 06** — Heaps (concept level); Transformer session 1: scaled dot-product attention
- **Day 05** — CS50P exceptions, dict/tuple/set practice, finished *AI Engineering* (Chip Huyen), hard LeetCode explored at intuition level
- **Day 04** — Kleppmann Ch.2 continued, three-way partition LeetCode
- **Day 03** — AWS Glue ETL, Kleppmann Ch.2, transformer paper — vectors and embeddings
- **Day 02** — GPT-2 internals, transformer components
- **Day 01** — CS50P, first LeetCode problems

---

## Up Next

- **Paper 2 of 7:** GPT-3 — *Language Models are Few-Shot Learners* (Brown et al., 2020). What happens when the architecture above is scaled ~1000×.
- Revisit the training loop with real backpropagation instead of numerical gradients.

---

## About

Penetration tester making a deliberate career pivot into AI/ML engineering. No formal math background, some Python experience, strong systems-thinking instinct from security work.

Every commit is a real session. Every problem log shows the actual mistakes.

**GitHub:** [Silverprompter](https://github.com/Silverprompter) | **Goal:** Anthropic AI/ML Engineer
