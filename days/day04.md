# Daily Log — June 8, 2026

## LeetCode

**Problem:** Partition Array According to Given Pivot  
**Difficulty:** Medium  
**Pattern:** Three-way partition

### Approach
Split the array into three separate buckets — less than pivot, equal to pivot, greater than pivot — while preserving the relative order of each group. Combine the buckets at the end.

### Solution
```python
def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
    less = []
    equal = []
    greater = []
    for x in nums:
        if x > pivot:
            greater.append(x)
        elif x < pivot:
            less.append(x)
        else:
            equal.append(x)
    return less + equal + greater
```

### Efficiency
- Time: **O(n)** — one pass through the array
- Space: **O(n)** — three lists holding all n elements

### Key lesson
Separating concerns makes the solution clean — collect first, combine last. No swapping, no index tracking. The tradeoff is extra space, but the clarity is worth it in an interview setting.

### Pattern connection
This is a variant of the **Dutch National Flag** problem — a classic three-way partition. Same idea: less, equal, greater.

---

## Designing Data-Intensive Applications — Chapter 2 (continued)

**Author:** Martin Kleppmann  
**Topic:** Data models and query languages

Continued reading through Chapter 2. The core theme is that data model choice shapes everything downstream — querying, scaling, and how relationships between data are expressed.

**Models covered so far:**
- **Relational model** — data in tables, relationships via joins
- **Document model** — data stored as self-contained documents, better for nested or hierarchical data
- **Graph model** — best for highly connected data where relationships are as important as the data itself

**Key takeaway:** there is no universally better model. The right choice depends entirely on the shape of your data and how you need to query it.

---

## Patterns and Connections Today

- Three-way partition → same bucket thinking as frequency maps and grouping problems
- Kleppmann data models → connects directly to the KV store diagnostic coming in Phase 0
- Relational vs document vs graph → useful mental model for thinking about how training data and model outputs get stored at scale

---

## Repo Updates Needed
- [ ] Add this log to `/logs` folder
- [ ] Update README weekly log row for Week 1
