# Daily Log — Day 5

## LeetCode

**Problem:** Maximum Total Value of K Distinct Subarrays  
**Difficulty:** Hard  
**Status:** Explored at intuition level — full solution parked

### What I learned
- The value of a subarray is `max - min`, so the most valuable subarrays are the ones containing both the global max and global min
- Counting distinct subarrays that contain two required positions: count valid start points (everything at or before the leftmost required position) times valid end points (everything at or after the rightmost required position)
- The `+1` in counting formulas comes from counting positions inclusively from zero
- The full solution requires a priority queue (heap) — a concept queued up for a future session

Parked the full implementation intentionally. Will revisit at a high level once heaps are covered.

---

## CS50P — Exceptions

Watched the CS50P lecture on exceptions — how Python handles errors at runtime:

- **try/except** — attempt risky code, catch the failure instead of crashing
- **ValueError, NameError** — different error types signal different problems
- **else and finally** — code that runs when no exception occurs, and code that runs no matter what
- **raise** — deliberately triggering an exception when something invalid happens

---

## Python Practice — Data Structures

Finished a code piece working with three core structures:

- **dict** — key-value lookups
- **tuple** — ordered, immutable groupings
- **set** — unique values, no duplicates

---

## Book Finished ✅

**AI Engineering** — Chip Huyen (O'Reilly, 2025)

Covers the process of building applications on top of foundation models — how AI engineering differs from traditional ML engineering, the new AI stack, evaluation pipelines, and inference optimization.

Fun connection: Chip Huyen also wrote *Designing Machine Learning Systems*, which is already on my reading list.

---

## Repo Updates Needed
- [ ] Add this log to `/logs` folder
- [ ] Update README daily log
- [ ] Mark AI Engineering as finished in books table
