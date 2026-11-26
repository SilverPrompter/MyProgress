# Day 18

*November 25, 2026 · Back after 26 days away · Block A done (cards + Rust); Block B (step 1) moves to Day 19*

---

## The plan changed today

Khan Academy results showed the math gap is wider than I thought: I completed Precalculus Unit 7 (Matrices) but not well, and couldn't start the first three questions of the Unit 6 (Vectors) test. So the next stretch is built around **one goal: build backprop from scratch.** Each piece of math gets learned exactly when that build needs it, then coded in Python the same day:

| Step | Math | Python I'll write |
|---|---|---|
| 1. A neuron is a dot product | vectors, dot product | `dot()`, `neuron()` with loops, then NumPy |
| 2. A layer is a matrix multiply | matrices | `matmul()` with nested loops, shape checks |
| 3. The loss | logarithms | `softmax()`, `cross_entropy()` |
| 4. Training follows the slope | derivatives | numerical derivative, gradient descent |
| 5. Backprop is the chain rule | chain rule, partial derivatives | hand-traced gradients in code |
| 6. Build it | all of it | a scalar autograd engine |

Longer-term: **applying to jobs by July 2027**, with a public portfolio. After backprop: my own GPT and tokenizer, then an "architecture zoo" that implements the techniques modern open models use (RoPE, GQA, mixture of experts, multi-head latent attention, linear attention, attention residuals), each built from its paper and measured.

---

## Review: 15 cards

After 26 days away: **4 right, 3 shaky, 8 wrong.** Expected. The useful part was relearning three things in the same session:

### Dot product: finally fixed

I'd been doing an **outer product** (each number times the *whole* other vector), then adding rows. Then I was **adding** down the columns instead of multiplying. What fixed it was four plain steps:

```
[3, 2] · [4, 5]

1. line them up:     3   2
                     4   5
2. first numbers:    3 × 4 = 12
3. second numbers:   2 × 5 = 10
4. add:              12 + 10 = 22
```

> **Down the column = multiply. Then across = add. One number out.**

### Adjacent pairs in a loop

`range(len(nums) - 1)` because **`i` is where each pair starts**, and the last pair starts one before the end. 6 items → 5 pairs → `i` goes 0 to 4.

### Cross-entropy, two steps

`loss = −ln(p)`: look up ln(p), then flip the sign. p = 1 → 0, p = 0.5 → 0.69, p = 0.1 → 2.3. Then on my own: p = 0.01, ln ≈ −4.6 → **loss 4.6** (badly wrong, big penalty).

### Also

- **Powers of 2 by doubling:** 2, 4, 8, 16, **32**, 64… I wrote 2⁵ = 64 and 2² = 8. New card.
- **Greedy algorithm** = take the best-looking step, never reconsider. Not always optimal (coins 1, 3, 4 making 6: greedy uses 3 coins, best is 2). Check it with brute-force differential testing.
- **Forward pass vs. backprop:** the forward pass happens in training *and* use; backprop only in training. My second miss on this.
- **Numerical gradient:** nudge each weight, run the model twice, compare. 2 runs per weight is impossible for billions of weights.

---

## Rust: chapter 2, the guessing game ✅

Built it by typing it myself. Check questions, 3 of 3 right:
- `mut`: variables can't change by default; without `mut` it won't compile
- the `rand` crate goes in `Cargo.toml` under `[dependencies]`
- Rust has **no exceptions**: `parse()` returns a `Result` (`Ok` / `Err`), handled with `match`:

```rust
let guess: u32 = match guess.trim().parse() {
    Ok(num) => num,
    Err(_) => continue,
};
```

---

## Numbers

- Cards: 15 reviewed (4 ✅, 3 🟡, 8 ❌), 4 new (powers of 2, greedy, `mut`, `Result`/`match`)
- Relearned to a correct fresh answer in-session: 3 (dot product, adjacent pairs, cross-entropy)
- Rust: chapters 1–2 done

## Next: Day 19

- Cards (today's misses come back first)
- **Step 1: a neuron is a dot product**, then code `dot()` and `neuron()` from a blank file into `builds/01-neuron/`
- Rust chapter 3

## Repo updates

- [x] `days/day_18.md`
- [x] README: Day 18 line + the backprop path
