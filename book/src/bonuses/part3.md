# Bonus challenges

## Challenge I: Advanced softmax implementations

*This challenge extends [Puzzle 16: Softmax Op](../puzzle_16/puzzle_16.md)*

Here are some advanced challenges to extend your softmax implementation:

### 1. Large-scale softmax: Handling `TPB < SIZE`

When the input size exceeds the number of threads per block (`TPB < SIZE`), our current implementation fails because a single block cannot process the entire array. Two approaches to solve this:

#### 1.1 Buffer reduction

- Store block-level results (max and sum) in device memory
- Use a second kernel to perform reduction across these intermediate results
- Implement a final normalization pass that uses the global max and sum

#### 1.2 Two-pass softmax

- First pass: Each block calculates its local max value
- Synchronize and compute global max
- Second pass: Calculate \\(e^{x-max}\\) and local sum
- Synchronize and compute global sum
- Final pass: Normalize using global sum

### 2. Batched softmax

Implement softmax for a batch of vectors (2D input tensor) with these variants:
- Row-wise softmax: Apply softmax independently to each row
- Column-wise softmax: Apply softmax independently to each column
- Compare performance differences between these implementations
