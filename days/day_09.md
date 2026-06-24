# Day 09

## LeetCode — Number Complement (Bit Manipulation)

**Problem:** Flip all the bits in an integer's binary representation and return the result.  
**Difficulty:** Easy  
**Patterns:** String transformation + bitmask construction with XOR

### Approach 1 — String transformation
Convert to binary, flip each character, convert back.

```python
def bitwiseComplement(self, n: int) -> int:
    bits = bin(n)[2:]
    result = ""
    for x in bits:
        if x == "1":
            result += "0"
        else:
            result += "1"
    return int(result, 2)
```

- Time: **O(b)** — one pass over the bits
- Space: **O(b)** — builds a result string

### Approach 2 — Bitmask + XOR (no strings)
Build a mask of all 1s matching n's bit length, then XOR.

```python
def bitwiseComplement(self, n: int) -> int:
    mask = 1
    while mask < n:
        mask = (mask << 1) | 1
    return n ^ mask
```

- Time: **O(log n)** — one step per bit
- Space: **O(1)** — pure integer math, no strings

### What I learned
- **XOR against 1 always flips a bit** — `0^1=1`, `1^1=0`. This is the core flip mechanism.
- **`<<` left shift** grows a number by one bit, padding a 0 on the right.
- **`| 1` (OR)** sets that new rightmost bit to 1.
- Combined, `(mask << 1) | 1` grows a string of all 1s: `1 → 11 → 111 → 1111`.
- The mask must match n's bit length exactly — too short and high bits don't flip, too long and we flip phantom bits. The loop auto-sizes it.
- Edge case `n = 0`: loop never runs, mask stays 1, `0 ^ 1 = 1` — correct.

These tricks (bitmasks, XOR flips, shifts) map directly onto security work — permissions, flags, crypto, network protocols.

---

## Repo Updates Needed
- [ ] Add this log to `/logs` folder as `day_09.md`
- [ ] Update README Recent Activity
- [ ] Add solution to `/leetcode`
