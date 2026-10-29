# Day 14

*October 27–28, 2026 · Re-entry session after 29 days away (last session: Day 13, Sept 28)*

---

## Format change

Starting today, sessions run as a conversation with Claude: spaced-repetition review, teaching, building, then a written log. A full program now sits behind this repo: a weekly rhythm, a 6-stage paper roadmap, a math track, a LeetCode pattern ladder and timed systems drills.

---

## Re-entry: rebuilding the transformer from memory

After a gap, rebuild the map before learning anything new. The test: explain the transformer as a **chain of problems**, where each component exists to fix something the previous one broke. No notes, no Day 11 log.

### First cold attempt

> words → numbers → vectors with dimensions → positions → decoders with feed-forward → normalization → backpropagation? → logits → tokens → output

**Right:** embeddings, positions, feed-forward, normalization, logits → tokens, all in the correct order.

**Missing:** attention, the most important piece. Without it, every word is processed alone.

**Wrong:** backpropagation doesn't belong in that list. The list is the **forward pass**, which runs every time the model reads text. Backprop happens **only during training**.

> Forward pass = taking the exam. Backprop = getting it graded and studying the mistakes. A deployed model only takes exams. Nobody grades, and the weights are frozen.

**Bigger lesson:** I gave the *pipeline* (what happens) instead of the *chain of problems* (why each piece exists). Memorizing the order of the parts isn't understanding. Knowing what breaks when you pull one out is.

---

## The chain, rebuilt link by link

| Problem | Fix | Recalled cold? |
|---------|-----|----------------|
| Words must become numbers | Embeddings | ✅ |
| Embeddings are fixed: "bank" is the same vector in "river bank" and "bank account" | **Attention**: rebuild each word from its context | ✅ |
| How much should each word attend to each other word? | **Dot product** = similarity (arrows pointing the same way) | ❌ |
| Raw scores can be any number, even negative | **Softmax**: positive, sums to 1 | 🟡 had the idea, called it "feed-forward" |
| One vector can't ask, advertise and deliver at once | **Q / K / V** | 🟡 named them, couldn't say why |
| Attention is order-blind | **Positional encoding**, added once at the start | 🟡 knew order matters, got the mechanism backwards |

About half recalled cold. That's what a 29-day gap costs. Every miss is now a review card.

### Why Q, K and V must be separate (the one I couldn't explain)

With one vector per word, the only question "it" can ask is *"who looks like me?"* That's the wrong question. In "The cat sat on the mat. **It** was tired," cat and mat are both short concrete nouns with nearby embeddings, so similarity alone can't pick.

W_Q and W_K are two different learned lenses:
- **W_Q** turns "it" into a *request*: "looking for something animate that could be tired"
- **W_K** turns each word into an *advertisement*: cat → "animate, subject"; mat → "inanimate, location"

The dot product now compares *what I'm looking for* against *what you're offering*.

**Contractor version:** a homeowner posts "need a licensed electrician" (**Query**). Each contractor's listing says what they do (**Key**). What shows up on site is the actual work (**Value**). With only one card per person, the homeowner would search for people whose card looks like *theirs* and find other homeowners.

> **Q and K decide where attention goes. V decides what gets delivered when it arrives.**

### Why attention is order-blind (the one I got backwards)

My answer: shuffling the sentence *changes* cat's output. Wrong. It **doesn't change at all**, and that's the problem.

> Attention's output is a weighted sum, and a sum doesn't care about order: **2 + 3 + 5 = 5 + 3 + 2**.

"Cat" compares itself to *the*, *cat* and *sat*. Shuffle the sentence and it's comparing itself to the same three words, getting the same scores and the same blend. Day 8 proved this in code. Today I learned it hadn't stuck without the code.

---

## The map: the transformer as a building

Most of today's confusion came from getting the parts without the structure they hang on.

```
EXIT      final norm → linear → logits → softmax → pick a token   (once)
          ──────────────────────────────────────────────
FLOOR N   [meeting room: attention] → [office: feed-forward]
  ...       identical floors, each with its own weights
FLOOR 1   [meeting room: attention] → [office: feed-forward]
          ──────────────────────────────────────────────
LOBBY     tokens → embeddings → + positional encoding            (once)
```

- **Transformer** = the whole building. **Block / layer** = one floor. In GPT-style models the stack of floors is the decoder.
- **Positional encoding lives in the lobby**, not inside the blocks. My instinct here was right.
- **Meeting room (attention):** words talk to each other. Q·K scores → softmax → **blend the values**.
- **Office (feed-forward):** each word thinks *alone* about what it heard. It never looks at other words.
- **Exit:** the last position's vector becomes one score per vocabulary word (logits), then softmax → probabilities → next token.

### Teach-back, and the one correction

My explanation back: lobby ✅, attention compares words ✅, exit with logits and softmax ✅. One mix-up: I put the *blending of values* in the feed-forward. It happens in **attention**. The feed-forward only transforms each word's own vector.

---

## Mix-ups caught today

| I said / thought | Actually |
|------------------|----------|
| Backprop is a step in the pipeline | Training only. The forward pass has no backprop |
| Scores → probabilities is the "feed-forward" | That's **softmax**, inside attention |
| Shuffling changes cat's output | It changes **nothing**. That's the bug |
| Values get blended in the feed-forward | Blending happens in **attention** |
| Positional encoding lives inside the decoder blocks? | Lobby, once, before any block |

---

## Reading comprehension lesson

Details arrived before structure, so they had nowhere to land. The fix for every paper from now on: **get the map before the parts.** First pass = headings, figures and conclusion only. Sketch the boxes and arrows. Then read for detail. (Pass 1 of Keshav's *How to Read a Paper*.)

---

## Connections

- Q / K / V ↔ job posting / contractor listing / the work delivered
- Order-blindness ↔ addition is commutative
- Building map ↔ Keshav pass 1: structure first, then contents
- Re-entry ↔ Day 11, which also opened by rebuilding the chain after a break. Gaps cost about half the recall each time. Consistency is the real bottleneck.

---

## Next: Day 15

- Open with review cards, including today's 9 misses and new cards
- Stage 1 begins: Keshav, *How to Read a Paper* (10 min), then pass 1 of Rumelhart, Hinton & Williams (1986), *Learning representations by back-propagating errors*
- First build target: a scalar autograd engine, so real backprop replaces the numerical gradients from Day 13

---

## Repo updates

- [ ] Add this log as `days/day_14.md`
- [ ] README Recent Activity: **Day 14**: re-entry after 29 days. Rebuilt the transformer from memory (~half recalled cold); fixed five mix-ups (backprop vs. forward pass, softmax vs. feed-forward, order-blindness, where values blend, where positional encoding lives); new "transformer as a building" map
- [ ] README Up Next: replace the GPT-3 line with **Stage 1: real backpropagation** (Rumelhart et al., 1986 → scalar autograd engine)
