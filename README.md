# Zero to ML Engineer — A Learning Journey

> From penetration tester to AI/ML engineer. Goal: Anthropic. Timeline: 17 weeks.

---

## Progress Overview

| Phase | Focus | Progress |
|-------|-------|----------|
| Foundations | Python + CS vocabulary | 🟡 `████░░░░░░` 40% |
| 0 | Python stdlib + KV diagnostic | ⬜ `░░░░░░░░░░` 0% |
| 1 | System building drills | ⬜ `░░░░░░░░░░` 0% |
| 2 | Public eval harness + blog | ⬜ `░░░░░░░░░░` 0% |
| 3 | System design + Kleppmann | ⬜ `░░░░░░░░░░` 0% |
| 4 | AI safety + STAR method | ⬜ `░░░░░░░░░░` 0% |
| 5 | Mock interviews + apply | ⬜ `░░░░░░░░░░` 0% |

**Parallel track:** GPT-2 internals study throughout all phases.

---

## What I'm Using

**Courses**
- Harvard CS50P — Python + CS fundamentals
- Khan Academy — Mathematics (working up to linear algebra and calculus)

**Video**
- 3Blue1Brown — Neural networks, linear algebra, calculus series
- Computerphile — CS concepts (hash tables, algorithms)

**Books**
- *Designing Data-Intensive Applications* — Martin Kleppmann (Phase 3)

**Practice**
- Daily LeetCode — learning-first workflow, one problem per day
- GPT-2 internals — full transformer stack from scratch in code

**Research Papers**
- Anthropic alignment and interpretability papers (Phase 4)

---

## Daily LeetCode Log

| # | Problem | Pattern | Difficulty |
|---|---------|---------|------------|
| 01 | Robot Return to Origin | Running counter balance | Easy |
| 02 | Check if Strings Can Be Made Equal | Group and compare | Easy |
| 03 | Matrix Cyclic Shift | Cyclic transformation | Medium |
| 04 | Grid Equal Sum Partition | Prefix sum | Medium |
| 05 | Flip Submatrix | Two pointer reversal | Medium |
| 06 | Biggest Rhombus Sums | Simulation + deduplication | Medium |
| 07 | Find Missing Binary String | Cantor's diagonal argument | Medium |

---

## Patterns Learned

- **Running counter balance** — initialize, modify in loop, check final state
- **Group and compare** — split into independent groups, check equivalence
- **Cyclic transformation** — use `k % n` for effective shift, slice to rotate
- **Prefix sum** — running total checked against a target at each step
- **Two pointer reversal** — one pointer each end, moving inward
- **Simulation + deduplication** — simulate every case, set handles duplicates
- **Cantor's diagonal argument** — construct a value guaranteed to differ from every item

---

## GPT-2 Internals Progress

- [x] Token and positional embeddings
- [x] Layer normalization
- [x] GELU activation
- [x] Multi-head attention — Query, Key, Value mechanics
- [x] Feedforward networks
- [x] Residual connections
- [x] Output head and text generation loop
- [x] Parameter counting
- [ ] Training loop and backpropagation
- [ ] Loss functions and cross entropy

---

## Weekly Log

| Week | What I Covered |
|------|---------------|
| Week 1 | CS50P Week 0-1, rock-paper-scissors, first LeetCode problems |
| Week 2 | GPT-2 internals, transformer components, LeetCode patterns |
| Week 3 | Medium LeetCode problems, Cantor's diagonal, repo setup |

---

## Blog

- [01 — From Breaking Systems to Building Them](blog/01_from_security_to_ml.md)

---

## About

Penetration tester making a deliberate career pivot into AI/ML engineering. No formal math background, some Python experience, strong systems-thinking instinct from security work.

Every commit is a real session. Every problem log shows the actual mistakes.

**GitHub:** [Silverprompter](https://github.com/Silverprompter) | **Goal:** Anthropic AI/ML Engineer
