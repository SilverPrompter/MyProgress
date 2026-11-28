# Day 20

*November 27, 2026 · Step 2 of the backprop path: a layer is a matrix multiply · Second portfolio build*

---

## Review: 15 cards

**6 right, 4 shaky, 5 wrong.** Almost all were brand-new cards from Day 19 on their first recall. The better sign: **four cards that started at "not sure" or "no clue" ended with me deriving the answer** through a few guided questions:

- **Why a dot product needs one loop:** the positions line up (0 with 0, 1 with 1), so one position number is enough. Two nested loops pair *every* item with *every* item.
- **The last `i` when comparing neighbors (6 items):** `nums[6]` doesn't exist, so `i + 1 ≤ 5`, so `i ≤ 4`. Derived, not memorized.
- **The length check:** compare `len(a)` and `len(b)` before the loop, and `raise ValueError(...)`. (`ValueError` = bad inputs; `IndexError` is the crash it prevents. Capitals matter.)
- **Rust `Result`:** `parse()` hands back `Ok(value)` or `Err(...)`, and `match` decides; `Err(_) => continue` asks again instead of crashing.

**Sign slip, root found:** I wrote the products as `-3 2` with no `+`, then read the 2 as negative. Habit: **always write the `+`**: `−3 + 2 = −1`.

---

## ML step: a layer is a matrix multiply

A layer = **many neurons reading the same input**, each with its own weights. Stack each neuron's weights as a **column** of `W`:

```
            A     B     C
W = [[ 0.5,   2,    1 ],     ← weights for input 1
     [ -1,  -0.5,   1 ]]     ← weights for input 2
```

`x @ W` = the dot product of `x` with **every column**: it's still just two vectors at a time, repeated once per neuron.

```
x (1, 2)  @  W (2, 3)  →  (1, 3)      one output per neuron
W shape = (inputs, neurons)
```

By hand, x = [2, 3], b = [1, 0.5, 0]:

| | A | B | C |
|---|---|---|---|
| x · column | −2 | 2.5 | 5 |
| + bias | −1 | 3 | 5 |
| ReLU | **0** | **3** | **5** |

The shape `(1, 2)` describes `x`'s size (1 row, 2 numbers), not its contents. That's the same idea as the Day 19 shape-notation fix.

---

## Python build: `builds/02-layer/`

Built on my own `dot()` and `relu()`:

```python
def column(w, j):
    nums = []
    for x in w:
        nums.append(x[j])
    return nums

def layer(x, W, b):
    outputs = []
    for j in range(len(W[0])):
        w_j = column(W, j)
        z = dot(x, w_j) + b[j]
        outputs.append(relu(z))
    return outputs
```

Output `[0, 3.0, 5]`, matching the hand calculation.

**Debugging I did:**
- `for x in W` gives whole **rows**, so item `j` is `x[j]` (not `W[x]`)
- `return` inside the loop quit after the first row, in `column()` and then in `dot()`. I found the second one from the clue that **every wrong output equaled just the first product plus the bias**.
- An "unresolved reference W" error: Python reads top to bottom, so `W` has to be defined before it's used.

---

## Rust: chapter 3.3–3.5

- No semicolon on a function's last line = an **expression** = its value is returned. Adding `;` makes it a statement → **compile error** (not a panic).
- `if` is an expression: `let size = if n < 5 { "small" } else { "big" };`
- `1..4` → 1, 2, 3 (end excluded, like Python's `range`); `1..=4` includes 4.

---

## Decisions

- **AWS ML Engineer Associate:** book the exam for mid-January; 45 min of practice questions on Saturdays.
- **Pace:** the card data (27% → 47% → 40%) says the current pace is hard enough. Automatic speed-up when card accuracy ≥ 75% and builds need no fill-in-the-blanks help, for a week.

## Numbers

- Cards: 15 (6 ✅ 4 🟡 5 ❌), 8 new (layer concepts + Rust)
- Builds: 2 total (`01-neuron`, `02-layer`)
- Rust: ch. 1–3.5

## Next: Day 21 (Saturday, light day)

- Cards
- Rust ch. 3 wrap-up / start ch. 4 (ownership)
- AWS practice questions (45 min)
