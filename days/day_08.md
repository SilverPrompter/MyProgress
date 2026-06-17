# Day 08

## LeetCode — Two Matrix Problems

### Problem 1: Determine Whether Matrix Can Be Obtained By Rotation
**Difficulty:** Easy  
**Pattern:** Matrix rotation (transpose + reverse)

Rotate a square grid in 90° increments and check if it matches a target. A square has only 4 distinct rotations, so check all four.

```python
def findRotation(self, mat: List[List[int]], target: List[List[int]]) -> bool:
    for _ in range(4):
        if mat == target:
            return True
        mat = [list(row)[::-1] for row in zip(*mat)]
    return False
```

**Key takeaway:** a 90° rotation = transpose (`zip(*mat)`) + reverse each row (`[::-1]`). The order can be swapped (reverse-then-transpose works too) — same result. Learned `zip(*grid)` as the transpose trick, and method chaining like `list(row)[::-1]`.

- Time: **O(n²)** — constant 4 rotations, each touches every cell
- Space: **O(n²)** — new grid per rotation

---

### Problem 2: Largest Submatrix With Rearrangements
**Difficulty:** Medium  
**Pattern:** Histogram heights + greedy sorting

Columns can be reordered freely. Find the largest all-1s rectangle. The insight: turn each column into a stacked height of consecutive 1s, then sort each row's heights descending. Because columns reorder freely, sorting clusters tall stacks together.

```python
def largestSubmatrix(self, matrix: List[List[int]]) -> int:
    m = len(matrix)
    n = len(matrix[0])
    for r in range(1, m):
        for c in range(n):
            if matrix[r][c] == 1:
                matrix[r][c] += matrix[r-1][c]
    best = 0
    for row in matrix:
        good = sorted(row, reverse=True)
        for i, h in enumerate(good):
            area = h * (i + 1)
            best = max(best, area)
    return best
```

**Key takeaway:** after sorting descending, everything to the left of a bar is at least as tall, so width = position + 1 and area = `height × (position + 1)`. The height-stacking transformation is the move that unlocks the whole problem.

- Time: **O(m × n log n)** — sorting each row dominates
- Space: **O(1)** in place (plus the sorted row)

---

## Transformer From Scratch — Sessions 3 & 4

Two sessions building toward a complete Transformer. Every line written by hand in NumPy.

### Session 3 — Positional Encoding

**The problem it solves:** attention is order-blind. Proved it directly — ran attention on "The cat sat" and the shuffled "sat cat The" and the middle word came out identical. The mechanism sees a bag of words, not a sequence. That breaks language ("dog bites man" ≠ "man bites dog").

**The fix:** stamp each position with a unique fingerprint built from sine and cosine waves, then **add** it to the word vectors before attention. Position is now baked into the input.

### Session 4 — Encoder Block

Assembled a real Transformer encoder block:

```
words in
  → attention (words share information)   + residual + layer_norm
  → feed-forward (each word thinks alone)  + residual + layer_norm
words out
```

Four ingredients working together: attention, feed-forward, residual connections, layer norm. The block takes shape `(3,4)` in and gives `(3,4)` out — **same shape on purpose**, so blocks can stack. Stacking is how you get a deep model.

**Bugs caught during the build:**
- `(std - 1e-9)` should be `(std + 1e-9)` — the epsilon prevents divide-by-zero; subtracting could cause it
- `att_out` vs `attn_out` — variable name mismatch
- Q/K/V order swap — passed `(W_Q, W_V, W_K)` but attention expects `(Q, K, V)`
- FFN shrink used `expanded @ W2` instead of `activated @ W2` — would have skipped the ReLU entirely, collapsing the feed-forward into a straight-line multiply

---

## Patterns and Connections Today

- Residual connections ↔ ResNet skip connections ↔ the gradient vanishing problem from AdderNet — same core idea across papers
- Layer norm ↔ BatchNorm from AdderNet — both rescale to a sane range, measured differently (per-word vs per-batch)
- Matrix rotation transpose trick ↔ the `zip` pairing of columns — same column-to-row movement
- Careful edge-case thinking and bug hunting — defensive-coding habit carried over from security work

---

## Repo Updates Needed
- [ ] Add this log to `/logs` folder as `day_08.md`
- [ ] Update README Recent Activity (newest on top)
- [ ] Add `03_positional_encoding.py` and `04_encoder_block.py` to repo (Sessions 3 & 4 of 7)
- [ ] Add both LeetCode solutions to `/leetcode`
