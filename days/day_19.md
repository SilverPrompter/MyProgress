# Day 19

*November 26, 2026 · Step 1 of the backprop path: a neuron is a dot product · First portfolio build*

---

## Day structure, fixed

Rust was coming before the ML step and Python, so when a day ran short, the most important work got dropped. New order, locked in: **cards → ML step → Python build → Rust → code reading/close.** If time runs short, cut from the bottom.

---

## Review: 15 cards

**7 right, 4 shaky, 4 wrong (47%, up from 27% yesterday).** Seven cards graduated, including three I'd missed before: forward pass vs. backprop, the exponent-rule proof, and cross-entropy on fresh numbers.

### Finds

- **Shape notation:** I kept answering "2 × 1" for `(3, 16) @ (16, 4)` because I read `(3, 16)` as a vector containing 3 and 16. It's a size: **3 rows, 16 columns**, like "12 × 10" describes a room. The answer is (3, 4).
- **`range(n)` never reaches n.** The last value is n − 1, so 6 items → `range(5)` → last `i` = 4.
- **Numerical gradient, computed by hand:** loss = w². At w = 5: (25.1001 − 24.9001) ÷ 0.02 = **10**. At w = 3 it's 6. The slope is always 2w — a derivative, found by experiment.
- **Powers of 2:** count by labeled doubling; 2⁸ = 256, 2¹⁰ = 1,024.

---

## ML step: a neuron is a dot product

```
neuron(x, w, b) = ReLU(w · x + b)
```

1. **Dot product:** weight each input, then add
2. **Add the bias:** how easily it fires
3. **ReLU:** on or off

This is the same `ReLU(m·x + b)` from Day 15, with many inputs instead of one.

| x | w | b | dot | + b | ReLU |
|---|---|---|---|---|---|
| [2, 3] | [0.5, −1] | 1 | −2 | −1 | **0** (off) |
| [1, 4] | [2, −0.5] | 0.5 | 0 | 0.5 | **0.5** (on: the inputs cancel and the bias tips it on) |
| [3, 1] | [−1, 2] | −0.5 | −1 | −1.5 | **0** (off) |

Slips I made: carrying a minus sign into the next column, and stopping before the final add. Rules: **each column has its own sign; a dot product isn't done until it's one number.**

---

## Python build: `builds/01-neuron/` (first portfolio piece)

**My first version was wrong in five ways**, and each one matched a slip from the cards:

| Bug | Why |
|---|---|
| Two nested loops | pairs every item with every item: an outer product, not a dot product |
| Multiplied `x * y` | those were **indices**, not the values `a[x]` |
| `range(len(a) - 1)` | skipped the last position (`- 1` is only for neighbors) |
| `if`/`else` on signs | unnecessary: `*` already handles signs |
| No `return` in `neuron` | printed `None` |

Fixed version, plus an input check I built by answering guided questions:

```python
def dot(a, b):
    if len(a) != len(b):
        raise ValueError("Lengths are not equal " + str(len(a)) + " vs " + str(len(b)))
    total = 0
    for i in range(len(a)):
        total += a[i] * b[i]
    return total
```

Then a **NumPy version**: `x @ w` replaces the whole loop. First I hit `TypeError: unsupported operand type(s) for @: 'list' and 'list'`: `@` needs NumPy arrays, so `x = np.array(x)` first. **Both versions agree on every test.**

---

## Rust: chapter 3.1–3.2

Typed every example, including the array-index program. Checks:
- **Shadowing in a block** prints 12 inside, then 6 outside: the inner shadow dies at `}`. (I had them swapped.)
- **Index 10 on a 5-item array** → panic at **run time**. Rust checks bounds instead of reading memory past the buffer (the out-of-bounds read bug class).
- **Shadowing vs. `mut`:** `mut` changes the value, same type; shadowing makes a new variable, any type.

Code: `rust/03-common-concepts/main.rs`

---

## How I want to be taught (feedback today)

When I'm stuck building, **walk me through the pieces I already know with questions** so I assemble the answer myself. The input-check extension was built that way, and I got every step right.

## Numbers

- Cards: 15 (7 ✅ 4 🟡 4 ❌), 10 new
- Builds: 1 (`01-neuron`, loop + NumPy versions, cross-checked)
- Rust: ch. 1–3.2

## Next: Day 20

- Cards (today's misses first)
- **Step 2: a layer is a matrix multiply** (starts from the shape notation fix)
- Build `matmul()` from a blank file
- Rust 3.3–3.5 (functions, comments, control flow)
