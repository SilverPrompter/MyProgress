# Rust Book ch. 3.1–3.2: variables and data types

Typed from the Book's examples: shadowing (including inside a block scope), integer and float types, arithmetic, booleans, chars, tuples (destructuring and `.0` access), arrays, and reading an index from the user.

What I checked afterward:
- The inner `{ }` shadow prints 12, the outer `x` prints 6: a shadow lives only inside its block.
- Entering an out-of-range index panics at **run time**: Rust bounds-checks every array access instead of reading memory past the end.
- `mut` changes a value (same type); shadowing creates a new variable (can change type).
