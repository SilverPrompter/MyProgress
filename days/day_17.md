# Day 17

*October 30, 2026 (closed out November 24, after a 25-day gap)*

---

## Review: 14 cards

Mostly cards last studied a month earlier, so this measures decay as much as learning.

| Result | Cards |
|---|---|
| ✅ | `nums[x+1]` vs. `nums[x] + 1` (the old recurring bug, now holding) |
| 🟡 | too many/too few heads, matrix shape rule, `sorted()` vs `.sort()`, which attention is masked, residual connections, forward pass vs. backprop, frozen vs. changing at inference, Transformers vs. RNNs (Recurrent Neural Networks) |
| ❌ | convergence (said confidently, and wrong), LayerNorm vs. BatchNorm, numerical gradients, why cross-entropy uses a log, the 4-D dot product |

## New things learned

- **Cross-attention:** the decoder's Q reads the encoder's K and V; not masked. Rule: *mask what you generate, never what you read.*
- **Convergence** = the loss stops improving. It doesn't mean the model is good.
- **Numerical gradients** cost 2 model runs **per weight**. Backprop gets every slope in one backward pass. The **learning rate** sets the step size (my GPT-2 fix: 0.004 → 0.0004 was an overshoot problem).
- **My idea: adjust groups of weights together to save runs.** That's essentially SPSA (Simultaneous Perturbation Stochastic Approximation, 1992) and MeZO (2023). Parked as a research question.
- **Logarithms:** the reverse of an exponent. `log₂(8) = 3`. Cross-entropy = `−ln(p_correct)`: 0 when certain and right, steep when confidently wrong.
- **When to stop vs. add:**

| Operation | Steps | `[1,2]` with `[3,4]` | NumPy |
|---|---|---|---|
| element-wise | multiply pairs, stop | `[3, 8]` | `a * b` |
| dot product | multiply pairs, **add** | `11` | `a @ b` |
| outer product | every × every | `[[3,4],[6,8]]` | `np.outer` |

## Code reading (Python)

Found both bugs (an off-by-one `range(len − 1)`; swapped `if`/`else` in a word counter) but predicted neither output exactly. The word counter actually crashes with `KeyError` on the first word. Lesson: **trace the first iteration from the real starting state.**

## Rust: chapter 1

Installed Rust, ran Hello World, made a Cargo project. `!` = macro; `cargo check` = fast "would it compile?"; compiled output goes to `target/`, not `src/`.

## Khan Academy (Oct 30 → Nov 24)

Precalculus Unit 7 (Matrices) completed, but weak. The Unit 6 (Vectors) test couldn't be started.

## Decision (Nov 25): backprop-driven, math on the fly

The ML ideas keep tripping over missing math. Instead of a separate math-first phase, the next stretch is driven by one goal, **building backprop**, and each math topic is learned exactly when that build needs it: vectors (a neuron is a dot product) → matrices (a layer is a matrix multiply) → logarithms (the loss) → derivatives → the chain rule (backprop itself). When something doesn't land, go back and review it, then continue. Checkpoint in 2 weeks: if math detours still dominate, switch to math-first.

## Repo updates

- [ ] Add this log as `days/day_17.md`
- [ ] README Recent Activity: **Day 17**: cross-attention, convergence, numerical gradients vs. backprop, logarithms, dot vs. element-wise vs. outer products; first Rust program. Decision: next stretch is backprop-driven, learning the math on the fly
