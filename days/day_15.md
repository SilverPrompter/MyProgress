# Day 15

*October 28–29, 2026 · First immersion day (cut short after Block A)*

---

## How the program changed today

- **Immersion days:** about 4 hours in three blocks. A: cards, code reading and Rust. B: the ML deep block. C: real codebases and the capstone. The floor stays at 20 minutes, and the rule is never to skip two days in a row.
- **Rust became a daily track**, and code reading (predict the output, find the bug, fix it, explain why) became a daily drill.
- **Math and algorithms come first for every paper:** a primer before reading, with hand-worked examples and hand traces.
- **SRS (Spaced Repetition System) is the backbone:** concepts get carded the moment they're learned, and new or missed cards go through learning steps (same-session relearn, then the next day) before entering the normal schedule.
- **Measured, not guessed:** every graded answer is logged with whether it was new or review, the result, the asks and the inferred confidence. Conclusions about how I learn need 5+ observations across 3+ topics over 2+ weeks.

---

## Review cards: 6 of 8

| Card | Topic | Result |
|------|-------|--------|
| C01 | What the feed-forward can do that attention can't | ❌ swapped them again (3rd time) |
| C15 | Why nonlinearity is required | ❌ → deep dive below |
| C42 | What an embedding is, geometrically | 🟡 dot product right, "matrix" wrong |
| C07 | Why subtracting the max doesn't change softmax | ❌ knew the rule, didn't recognize it |
| C08 | Mask before vs. after softmax | ✅ |
| C22 | Why multi-head attention | ❌ thought heads split the batch |

All six were reviews after a 29-day gap, so this mostly measures decay, not a baseline.

---

## Deep dive: ReLU (Rectified Linear Unit)

**Why nonlinearity:** without activations, layers collapse.

```
(x @ W1) @ W2  =  x @ (W1 @ W2)  =  x @ W     ← 100 layers = 1 matrix
```

**ReLU(x) = max(0, x).** A neuron computes `W2 × ReLU(m·x + b)`, and each neuron is **one hinge**:
- **b** moves the bend (it sits at x = −b/m)
- **m** (W1) sets the steepness, and moves the bend only as a side effect
- **W2** sets how much the total slope changes after the bend, and which direction

**Hinges add up.** The spike `f(x) = ReLU(x) − 2·ReLU(x−1) + ReLU(x−2)`:

| Region | Hinges active | Total slope |
|---|---|---|
| x < 0 | none | 0 |
| 0 to 1 | term 1 | +1 |
| 1 to 2 | terms 1 + 2 | +1 − 2 = −1 |
| x > 2 | all three | −1 + 1 = 0 |

Traced at x = 1.5: inside values 1.5, 0.5, −0.5 → ReLU gives 1.5, 0.5, 0 → × W2 gives 1.5, −1, 0 → **0.5**.

**Remove the ReLUs and the same numbers give** `x − 2(x−1) + (x−2) = 0` for every x. The lines cancel completely, which is the Day 8 bug made concrete.

**Many bends = an unsmoothed curve.** Four hinges (slopes 1 → 3 → 5 → 7) approximate the quadratic x². More hinges, smoother curve.

**In 3D (two inputs):** each neuron is a tilted sheet, and ReLU folds it along a crease. Summing neurons stacks folded sheets.

**Vocabulary fixed:** linear (constant slope) vs. quadratic (U shape, x²) vs. exponential (2^x, slope grows with height). "Nonlinear" covers all the curved ones. W2 scales linearly; compounding (exponential) only happens across many stacked layers, which is the vanishing / exploding gradient problem.

---

## Embeddings and the dot product

- **Embedding** = one vector = one point (or arrow from the origin). **Embedding matrix** = one row per token. **Dimensions** = how many numbers are in each vector.
- **Dot product** = multiply matching positions, then add. `a · b = |a||b|cos(angle)`: same direction → big, perpendicular → 0, opposite → negative.
- `[2,1] · [−1,2] = 0` (perpendicular). `[1,2,0,3] · [2,1,4,1] = 7`.
- **My own conclusion:** you can't graph 4+ dimensions, but the math works the same in any number of dimensions. **The limit is the picture, not the math.**
- Ways to picture more dimensions: pick a few dimensions, encode extras as color/size/motion, or PCA (Principal Component Analysis), which I've parked for the math track.

---

## Softmax

| Step | Job |
|---|---|
| Subtract the max | Safety only, prevents overflow (`e^1000 = inf`) |
| Exponentiate | Makes everything positive |
| Divide by the sum | Each value 0–1, all summing to 1 |

**Why subtracting the max changes nothing:** `e^(a−b) = e^a / e^b`. Every term is divided by the same e^max, top and bottom, so it cancels (like 6/8 = 3/4). I knew the exponent rule; the new context hid it.

**ReLU vs. softmax on negatives:** ReLU zeroes them (−1 and −50 both become 0). The exponent shrinks them but keeps the order (e⁻¹ = 0.368).

**Masking:** mask with −infinity **before** softmax. Zeroing after softmax leaves rows summing to less than 1 (`[0.731, 0]`).

---

## Multi-head attention

- Heads split the **dimensions** (d_model ÷ heads), not the context window and not the batch. Every head sees every word.
- **Why:** one softmax gives one blend per word, but a word needs several kinds of information at once (its subject, what describes it, its neighbor). Heads give several independent attention patterns for about the cost of one.
- **Who assigns head jobs:** nobody. Random initialization breaks symmetry (identical heads would stay identical forever). *Which jobs exist* is driven by the data; *which head number gets which job* is arbitrary.
- **Too many heads:** each gets too few dimensions. GPT-2 small = 768 ÷ 12 = 64 per head. *"Are Sixteen Heads Really Better than One?"* (2019): many heads can be removed with little loss.
- **Parameter vs. hyperparameter:** learned weights vs. designer-chosen settings (heads, layers, learning rate).
- **Hand-assigned heads** (syntax-guided, fixed, sparse/local patterns) can save compute or data, but you only get the patterns you thought of. **The Bitter Lesson** (Sutton, 2019): general methods that scale with compute and data beat hand-coded knowledge.

---

## Previewed: *Attention Residuals* (Kimi Team / Moonshot AI, March 2026)

Normal residuals add every earlier layer's output with a fixed weight of 1, which dilutes each layer as depth grows. Attention Residuals replace that sum with **softmax attention over earlier layers' outputs**: score → softmax → blend, the same three steps as word attention, aimed across depth instead of across words. The designer cuts the layers into about 8 blocks to save memory. Added to the roadmap right after ResNet for a full study.

**Depth vs. width:** stacked ReLU layers fold already-folded space, so deep networks get exponentially more linear pieces (Montúfar et al., 2014). The cost is sequential latency. Scaling laws (Kaplan, 2020): total parameters matter most, and shape only weakly.

**Attention's cost** grows with the square of the input length. Current directions: sparse attention, linear attention and SSMs (State Space Models, e.g. Mamba), and hybrids. **My take:** these feel like bandaids, and a new mechanism is needed. Parked as a research hypothesis to revisit after Stage 3.

---

## Numbers

- Cards reviewed: 6 (1 ✅, 1 🟡, 4 ❌)
- New cards created: **22** (C52–C73), carded the moment each concept landed
- Asks: about 4 on ReLU, 2 on embeddings / dimensions, 2 on Attention Residuals
- Self-generated insights: 3 (the limit is the picture not the math; bends make an unsmoothed curve; attention alternatives as bandaids)

---

## Next: Day 16

- **Longer review block (~30 min):** today's learning-step cards plus misses (capped at about 25, priority to learning and missed cards)
- Cards 7–8 roll over
- Code reading (Python) → Rust Book ch. 1 → Keshav's *How to Read a Paper* → the backprop math and algorithm primer

---

## Repo updates

- [ ] Add this log as `days/day_15.md`
- [ ] README Recent Activity: **Day 15**: first immersion day. Deep dives on ReLU (hinges, folding, why depth needs nonlinearity), embeddings and the dot product, softmax stability, and multi-head attention. Previewed *Attention Residuals* (2026). The program now has daily Rust, math-first paper primers and SRS learning steps
