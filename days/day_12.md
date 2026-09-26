# Day 12

## Deep Learning Foundations — Perceptron, Activations, CNNs

### The perceptron and where it came from

- **1943 — McCulloch & Pitts.** An artificial neuron built as a threshold logic unit: take weighted inputs, sum them, fire if the sum clears a threshold. No learning — weights were set by hand. The point was to show a neuron-like unit could compute logical functions.
- **1958 — Rosenblatt's perceptron.** The addition that mattered: a *learning rule*. Weights update themselves in response to errors. This is the ancestor of every trained network since.

The core unit hasn't changed: **inputs → weights → sum → activation → output.** Everything modern is that, stacked and scaled.

### Activation functions

Without an activation function, stacking layers is pointless — a chain of matrix multiplies collapses into a single matrix multiply, so a hundred layers has exactly the expressive power of one. The activation is the nonlinearity that makes depth mean something.

(Same reason the feed-forward bug from Transformer Session 4 mattered — multiplying by `expanded` instead of `activated` skipped the ReLU and collapsed the layer into a straight-line multiply.)

### CNNs — why they work on images

A dense layer treats every pixel as unrelated to its neighbors. Images don't work that way — meaning lives in *local* structure, and an edge is an edge wherever it appears. CNNs are built around both facts.

The pipeline:

1. **Convolution** — slide a small filter across the 2D input, computing a weighted sum at each position. The same filter is reused everywhere (weight sharing), so a feature detected in one corner is detected in all of them. Output is a feature map.
2. **Max pooling** — downsample by keeping only the strongest value in each small window. Shrinks the data, keeps the signal, and buys a little tolerance to small shifts in position.
3. **Flatten** — collapse the remaining 2D feature maps into a 1D vector so a dense (perceptron-style) layer can consume it.
4. **Dense layers** — the classifier on top, learning combinations of the features convolution found.
5. **Dropout** — randomly switch off a fraction of units during training so the network can't lean on any single path. A regularizer; off at inference time.
6. **Softmax** — final layer, turns raw scores into probabilities across the classes that sum to 1.

**Mental model:** convolution and pooling are a funnel that turns raw pixels into a compact set of "what's in here" features. The dense layers at the end are the ordinary perceptron doing the actual deciding — they just get to work on features instead of pixels.

---

## LeetCode — Check if Array Is Sorted and Rotated (retrieval attempt)

Re-attempted a problem solved earlier (originally logged as problem 08). Treated as a memory test rather than a new problem.

**Result: did not recall the pattern.** First instinct was to sort a copy and test rotations against it — a different, heavier approach. Rebuilt the original one-pass solution with significant scaffolding.

```python
def check(self, nums: List[int]) -> bool:
    count = 0
    flag = False
    for x in range(len(nums) - 1):
        if nums[x] > nums[x+1]:
            count += 1
    if nums[-1] <= nums[0]:
        flag = True
    if count == 0:
        return True
    elif count == 1 and flag:
        return True
    else:
        return False
```

- Time: **O(n)** — one pass
- Space: **O(1)** — two variables

**The logic:** a sorted array only ever goes up. Rotating it moves one chunk to the front, creating exactly one drop. Zero drops means never rotated (valid on its own). One drop is valid only if the last element is ≤ the first, so the tail can wrap around and meet the head. Two or more drops is impossible from any rotation.

### The real finding: one recurring bug, three appearances

Three separate stumbles across today and yesterday were the same root cause:

| Wrote | Meant |
|-------|-------|
| `nums[x] - 1` | `nums[x+1]` |
| `group[x+1]` | `group[x-1] + 1` |
| `range(len(nums))` | `range(len(nums) - 1)` |

**The rule:** arithmetic *inside* the brackets moves where you're looking. Arithmetic *outside* the brackets changes the value you found. `nums[x+1]` is the next element; `nums[x] + 1` is this element plus one.

The `range` case is the same idea wearing a different hat: when a loop compares **pairs**, the last valid start position is one short of the end.

Also flagged: `&&` is not Python — it's `and`. Worth locking in ahead of the eventual C++ work, where both spellings exist across the two languages.

**Next session opens with an index-expression drill** — reading `nums[...]` expressions against a fixed array until it's automatic — before any new problem.

---

## Connections

- Activation nonlinearity ↔ the Session 4 feed-forward bug (skipping ReLU collapses the layer)
- Weight sharing in convolution ↔ positional encoding: both inject structure the raw mechanism is blind to
- Dropout ↔ residual connections and layer norm: the family of tricks that make deep stacks trainable
- Spaced retrieval as a study method — re-solving an old problem cold exposed a gap that re-reading the solution never would have

---
