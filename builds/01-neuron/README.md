# 01 · A neuron from scratch

**Step 1 of the backprop path:** a neuron is a dot product.

A neuron takes inputs `x`, weights `w` and a bias `b`, and computes:

```
output = ReLU(w · x + b)
```

1. **Dot product:** multiply matching positions, then add → one number
2. **Add the bias:** sets how easily the neuron fires
3. **ReLU:** positive passes through, negative becomes 0 (off)

Two versions, both written from a blank file:
- `neuron()`: pure Python, a loop for the dot product
- `neuron_np()`: NumPy, where `x @ w` does the whole dot product

The two are checked against each other on the same inputs (a slow, obvious version verifying a fast one).

## Run

```bash
python neuron.py
```

## Results

Every test was worked by hand first, then checked against the code:

| Call | By hand | Code |
|---|---|---|
| `dot([3, 2], [4, 5])` | 3×4 + 2×5 = 22 | 22 ✅ |
| `neuron([2, 3], [0.5, -1], 1)` | −2 + 1 = −1 → off | 0 ✅ |
| `neuron([1, 4], [2, -0.5], 0.5)` | 0 + 0.5 = 0.5 → on | 0.5 ✅ |
| `neuron([3, 1], [-1, 2], -0.5)` | −1 − 0.5 = −1.5 → off | 0 ✅ |
| `dot([3, 2, 1], [3, 3])` | lengths differ | `ValueError: Lengths are not equal 3 vs 2` ✅ |
| `neuron_np(...)` on the three neuron cases | same as the loop version | 0, 0.5, 0 ✅ (agrees) |

## What I learned building it

- **My first version used two nested loops.** That pairs *every* item with *every* item (an outer product), not matching positions. A dot product needs **one** loop, with the same index `i` used for both lists.
- `range` gives **positions**, not values: the number is `a[i]`.
- No `- 1` in the range: that's only for comparing neighbors. Here every position is touched once.
- Multiplication already handles negative signs, so no `if` is needed; just add each product to the total.
- A function that doesn't `return` gives back `None`.
- Check inputs up front: mismatched lengths raise a clear `ValueError` before the loop starts, instead of crashing halfway through with an `IndexError`.
- NumPy: `a * b` multiplies the pairs and stops; `a @ b` multiplies and adds (the dot product). `@` only works on NumPy **arrays**, not plain lists (`TypeError: unsupported operand type(s) for @: 'list' and 'list'`), so convert first with `x = np.array(x)`.

## Next

- Step 2: a layer is a matrix multiply
