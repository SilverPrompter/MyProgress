# 02 · A layer from scratch

**Step 2 of the backprop path:** a layer is a matrix multiply.

A layer is many neurons reading the same input, each with its own weights and bias. Stack each neuron's weights as a **column** of `W`:

```
            A     B     C
W = [[ 0.5,   2,    1 ],     ← weights for input 1
     [ -1,  -0.5,   1 ]]     ← weights for input 2

W shape = (inputs, neurons) = (2, 3)
```

Then the whole layer is one matrix multiply, plus a bias per neuron, then ReLU on each:

```
layer(x) = ReLU(x @ W + b)        (1, 2) @ (2, 3) → (1, 3): one output per neuron
```

A matrix multiply is just the dot product rule run once per column: x · column A, x · column B, x · column C.

## The code

Pure Python, built on `dot()` and `relu()` from [`01-neuron`](../01-neuron):

- `column(W, j)`: collects neuron j's weights (item j of every row)
- `layer(x, W, b)`: for each neuron j, `relu(dot(x, column(W, j)) + b[j])`

## Results

| Step | By hand | Code |
|---|---|---|
| `column(W, 0)`, `column(W, 2)` | [0.5, −1], [1, 1] | ✅ |
| `x @ W` with x = [2, 3] | [−2, 2.5, 5] | |
| `+ b` with b = [1, 0.5, 0] | [−1, 3, 5] | |
| ReLU | [0, 3, 5] | `[0, 3.0, 5]` ✅ |

## Bugs I hit

- `return` indented **inside** the loop, in both `column()` and `dot()`: the function quit after the first item. Found from the clue that every wrong output equaled *only the first product plus the bias*.
- `for x in W` gives whole **rows**, not positions, so the item is `x[j]`, not `W[x]`.
- `NameError`-style "unresolved reference W": Python reads top to bottom, so `W` must be defined before the line that uses it.

## Next

NumPy version (`relu(x @ W + b)` in one line) checked against this one, then step 3: the loss.
