# Transformer

## Overview

A Transformer is built from repeated Transformer Blocks.

A Transformer Block combines:

1. Multi-Head Self-Attention
2. Residual Connection
3. Layer Normalization
4. Feed Forward Network
5. Residual Connection
6. Layer Normalization

Basic structure:

Input
  ↓
Multi-Head Self-Attention
  ↓
Residual Connection
  ↓
Layer Normalization
  ↓
Feed Forward Network
  ↓
Residual Connection
  ↓
Layer Normalization
  ↓
Output


---

# 1. Multi-Head Attention

Multi-Head Attention runs multiple attention mechanisms in parallel.

Each head can learn different relationships between tokens.

Basic idea:

Input
  ↓
Multiple Attention Heads
  ↓
Concatenate Head Outputs
  ↓
Multi-Head Output

For example:

- embedding_dimension = 8
- num_heads = 2
- head_dimension = 4

The outputs are concatenated:

Head 1 (4 dimensions) + Head 2 (4 dimensions)
→ 8 dimensions

### Self-Attention

For each token:

Q = XW_Q
K = XW_K
V = XW_V

Attention scores:

scores = QK^T

Scaled scores:

scaled_scores = QK^T / sqrt(d_k)

Attention weights:

weights = Softmax(scaled_scores)

Attention output:

Attention(Q,K,V) = weights × V

Attention allows tokens to exchange information with other tokens.

Key idea:

Attention = communication between tokens.


---

# 2. Feed Forward Network

The Feed Forward Network processes each token independently.

Formula:

FFN(X) = ReLU(XW_1 + b_1)W_2 + b_2

Typical shape flow:

X
(3, 8)
  ↓
W_1
(8, 16)
  ↓
Hidden
(3, 16)
  ↓
ReLU
(3, 16)
  ↓
W_2
(16, 8)
  ↓
Output
(3, 8)

The network expands and then contracts the representation:

8 → 16 → 8

The larger hidden dimension provides more space for transformations.

ReLU:

ReLU(x) = max(0, x)

ReLU introduces non-linearity.

Key idea:

Attention = communication

FFN = computation / transformation


---

# 3. Residual Connection

A residual connection adds the original input to the transformed output.

Formula:

Output = Input + Transformation(Input)

Example:

residual_1 = X + Attention(X)

The original representation is not discarded.

Benefits:

- Preserves information from previous layers
- Makes deep networks easier to train
- Provides a shortcut path for information and gradients

Transformer blocks use residual connections around both major sub-layers:

1. Attention
2. Feed Forward Network


---

# 4. Layer Normalization

Layer Normalization normalizes each token's representation independently.

For:

X.shape = (3, 8)

LayerNorm calculates statistics across the 8 dimensions of each token.

Mean:

mean = np.mean(
    X,
    axis=-1,
    keepdims=True
)

Variance:

variance = np.var(
    X,
    axis=-1,
    keepdims=True
)

Normalization:

normalized =
    (X - mean) /
    sqrt(variance + epsilon)

where:

epsilon = 1e-5

The small epsilon prevents division by zero and improves numerical stability.

Then:

output = gamma * normalized + beta

where:

- gamma = learnable scale parameter
- beta = learnable shift parameter

Initially:

gamma = 1
beta = 0

### NumPy Parameters

np.mean(X, axis=-1, keepdims=True)

- X: input array
- axis=-1: operate on the last dimension
- keepdims=True: keep the reduced dimension with size 1

np.var() uses the same parameters.

For X.shape = (3, 8):

axis=-1 means:

calculate statistics across the 8 dimensions of each token.

keepdims=True changes:

(3,) → (3, 1)

This makes broadcasting easier for operations such as:

X - mean


---

# 5. Transformer Block

The complete Transformer Block:

X
 ↓
Multi-Head Self-Attention
 ↓
Residual Connection
 ↓
LayerNorm
 ↓
Feed Forward Network
 ↓
Residual Connection
 ↓
LayerNorm
 ↓
Output

Mathematically:

Attention sub-layer:

X_1 = LayerNorm(
    X + Attention(X)
)

FFN sub-layer:

X_2 = LayerNorm(
    X_1 + FFN(X_1)
)

Final:

Output = X_2


---

# 6. Why Two Residual Connections?

There are two major transformations inside a Transformer Block:

1. Attention
2. FFN

Each transformation gets its own residual connection.

First:

X + Attention(X)

Second:

X_1 + FFN(X_1)

This allows the model to preserve previous information while learning new transformations.


---

# 7. Why Two Layer Normalizations?

Each major sub-layer is followed by normalization:

Attention
  ↓
Residual
  ↓
LayerNorm

FFN
  ↓
Residual
  ↓
LayerNorm

LayerNorm helps keep the representation at a stable scale throughout the network.


---

# 8. Shape Tracking

Example:

sequence_length = 3
embedding_dimension = 8
hidden_dimension = 16

Overall:

Input
(3, 8)
  ↓
Attention
(3, 8)
  ↓
Residual
(3, 8)
  ↓
LayerNorm
(3, 8)
  ↓
FFN
(3, 16)
  ↓
FFN Output
(3, 8)
  ↓
Residual
(3, 8)
  ↓
LayerNorm
(3, 8)
  ↓
Output
(3, 8)

The model dimension remains 8 throughout the block.

Only the internal FFN hidden dimension expands:

8 → 16 → 8


---

# 9. Conceptual Difference

### Attention

Allows tokens to interact with each other.

Example:

"I love Japanese"

The representation of "Japanese" can incorporate information from "I" and "love".

### FFN

Processes each token representation independently.

Attention answers:

"What information from other tokens is relevant?"

FFN answers:

"How should this token representation be transformed?"


---

# 10. Transformer Stack

A Transformer is usually composed of multiple Transformer Blocks.

Input Embeddings
      ↓
Transformer Block 1
      ↓
Transformer Block 2
      ↓
Transformer Block 3
      ↓
...
      ↓
Transformer Block N
      ↓
Output

Each block progressively transforms the token representations.

The output of one block becomes the input of the next block.


---

# 11. Learning vs Toy Implementation

The code in this lab is an educational NumPy implementation.

The weights are randomly initialized and are not trained.

For example:

W_Q = np.random.randn(...)
W_K = np.random.randn(...)
W_V = np.random.randn(...)

In a real Transformer:

1. Parameters are initialized
2. Input passes through the Transformer
3. The model produces predictions
4. Loss is calculated
5. Backpropagation calculates gradients
6. Parameters are updated
7. The process repeats

Therefore, the random values in this lab demonstrate the architecture, not learned language knowledge.


---

# 12. Summary

The core Transformer Block can be remembered as:

Input
  ↓
Attention
  ↓
Add Input
  ↓
LayerNorm
  ↓
FFN
  ↓
Add Previous Representation
  ↓
LayerNorm
  ↓
Output


Key concepts:

- Attention → token-to-token communication
- Multi-Head Attention → multiple attention patterns in parallel
- FFN → token-wise transformation
- Residual Connection → preserve information + improve gradient flow
- LayerNorm → stabilize representations
- Transformer Block → Attention + FFN + Residual + LayerNorm
- Multiple Blocks → deeper representation learning