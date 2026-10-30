# Day 16

*October 29–30, 2026 · Review-heavy day; ended at card 20 of 25 (relearn round, code reading and Rust not reached)*

---

## Review: 20 of 25 cards

All of these were yesterday's new cards on their second exposure, or yesterday's misses. Wobbles here are expected.

| Result | Count | Cards |
|---|---|---|
| ✅ right | 1 | hinges add up to a curve (graduated to box 1) |
| 🟡 shaky | 12 | context window vs. batch vs. dimensions, exponent rule, linear vs. quadratic vs. exponential, no-ReLU output, identical heads, softmax steps, ReLU vs. exponent on negatives, Bitter Lesson, embedding vs. matrix, multi-head attention, softmax's max, who decides head jobs |
| ❌ wrong | 7 | ReLU knobs (m, b, W2), attention vs. feed-forward, parameter vs. hyperparameter, dot product, spike trace, nonlinearity (all later understood) |

---

## The big find: the gaps are mostly math, not ML

Attention vs. feed-forward got swapped for the **4th time in 3 days**. Digging in, the root turned out to be **matrix multiplication**, which I'd been using since Day 6 without ever being taught how it works. When the matrices were removed, I got it right away:

```
Feed-forward:  new word 1 = f(word 1)                     ← word 1 alone
Attention:     new word 1 = 0.3 × word 0 + 0.7 × word 1   ← a blend
```

### Matrix multiplication, finally

```
(a, b) @ (b, c) → (a, c)      middles must match and disappear; outers survive
```

Each output cell = **row i of the left · column j of the right** (a dot product).

```
[[1,2,3],     @   [[1,0],      =   [[ 4,  5],
 [4,5,6]]          [0,1],           [10, 11]]
                   [1,1]]
cell [0,0] = 1×1 + 2×0 + 3×1 = 4
```

**The payoff:** rows of the left matrix never mix; only the shared middle dimension gets summed.

| | Rows (stay separate) | Middle (summed over) | Words mix? |
|---|---|---|---|
| Feed-forward `x @ W` | words | features | no |
| Attention `weights @ V` | words | **words** | **yes** |

Built a color-coded diagram with lines from each input number to the output cell it feeds. In the feed-forward, all lines into a word's output share one color; in attention, both words' colors arrive.

### Other math gaps found today

| Gap | Where it bit |
|---|---|
| Negative exponents: `2^(−1) = 1/2` | exponent rule card |
| Canceling a common factor: a share of a total doesn't change when everything shrinks equally | why subtracting the max leaves softmax unchanged |
| Keeping a negative sign through a multiply: `−2 × 0.5 = −1` | spike trace (got 2.5 instead of 0.5) |
| Dot product vs. outer product: multiply pairs, then **add** to get one number | `[2,1]·[−1,2] = 0` (did each element × the whole vector instead) |
| Base 1 as a test case: `1^x = 1` always, so it proves nothing | exponent check, exponential example |

**Program change:** algebra refresher (Khan Academy: exponents, rational expressions) plus matrix multiplication practice, before the backprop calculus.

---

## Concepts that clicked

- **The ReLU knobs**, via interactive sliders: **b** moves the bend, **m** sets the steepness (and nudges the bend, since it sits at x = −b/m), **W2** sets the direction and size of the slope change.
- **Why nonlinearity is needed:** stacked straight lines are always one straight line, and straight lines can't turn around. A "low, high, low" bump (like flagging logins that are too fast *or* too slow) is impossible for a line and exact for three ReLU hinges.
- **What heads split:** the **dimensions**, never the words or the batch, because every head must see every word. One head = one blend; several heads = several blends.
- **Convergence:** the loss stops improving and the weights settle. It doesn't mean the model is good, and big models are often stopped before converging.
- **Parameter vs. hyperparameter:** does training change it? Yes → parameter (weights). No → hyperparameter (number of heads, learning rate).
- **Softmax as a pipeline:** make it safe (subtract the max) → make it positive (exponentiate) → make it sum to 1 (divide).

---

## Numbers

- Cards reviewed: 20 (1 ✅, 12 🟡, 7 ❌)
- New cards: 4 (matrix multiplication shape rule, matrix multiply by hand, which dimension mixes, convergence)
- Biggest ask counts: nonlinearity (5 total), ReLU knobs (4 total), attention vs. feed-forward (4 today)
- Formats that resolved stuck points: interactive sliders, colored matrices with lines, a labeled pentest analogy with a chart

---

## Next: Day 17

- Review (capped at 25): learning cards first, including today's misses
- Relearn round at the end of the session
- Then the parts skipped today: code reading, Rust chapter 1

---

## Repo updates

- [ ] Add this log as `days/day_16.md`
- [ ] README Recent Activity: **Day 16**: review day that found the real bottleneck: math foundations (matrix multiplication, exponents, fractions), not ML concepts. Learned matrix multiplication properly; resolved the attention vs. feed-forward swap and why nonlinearity is needed
