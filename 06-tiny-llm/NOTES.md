# Tiny LLM

## 1. Overview

This stage builds a small language model architecture using PyTorch.

Previous stages implemented the main concepts with NumPy:

```text
01 Tokenization
       ↓
02 Embeddings
       ↓
03 Attention
       ↓
04 Transformer
       ↓
05 Inference
```

Stage 6 connects these concepts into one Tiny LLM:

```text
Token IDs
    ↓
Token Embedding
    ↓
Position Embedding
    ↓
Token + Position
    ↓
Transformer Block
    ↓
Contextualized Representation
    ↓
Language Model Head
    ↓
Logits
    ↓
Next Token
```

The model is an actual PyTorch neural network architecture, but it is not trained yet.

Training will be covered in Stage 7.

---

# 2. Tiny LLM Architecture

The Tiny LLM contains:

```text
TinyLLM
│
├── Token Embedding
│
├── Position Embedding
│
├── Multi-Head Self-Attention
│
├── LayerNorm
│
├── Feed Forward Network
│
├── LayerNorm
│
└── Language Model Head
```

Conceptually:

```text
Input Token IDs
       ↓
Token Embedding
       +
Position Embedding
       ↓
Transformer Block
       ↓
Language Model Head
       ↓
Logits
```

---

# 3. Configuration

Example configuration:

```python
vocabulary_size = 10
embedding_dimension = 8
hidden_dimension = 16
max_sequence_length = 10
num_heads = 2
```

Meaning:

```text
vocabulary_size
    = number of possible tokens

embedding_dimension
    = size of each token representation

hidden_dimension
    = size of the FFN intermediate representation

max_sequence_length
    = maximum sequence length supported by position embeddings

num_heads
    = number of attention heads
```

---

# 4. PyTorch `nn.Module`

The model inherits from:

```python
class TinyLLM(nn.Module):
```

`nn.Module` is the base class for PyTorch neural network models.

It provides:

- parameter management
- submodule management
- training/evaluation modes
- device management
- state dictionaries
- autograd integration

Example:

```python
model = TinyLLM(...)
```

PyTorch can then discover the model's parameters automatically.

---

# 5. Token Embedding

Token embedding is created with:

```python
self.token_embedding = nn.Embedding(
    vocabulary_size,
    embedding_dimension
)
```

For:

```text
vocabulary_size = 10
embedding_dimension = 8
```

the embedding matrix has shape:

```text
(10, 8)
```

Conceptually:

```text
Token ID
   ↓
Embedding Matrix
   ↓
Token Vector
```

Example:

```text
ID 2 → vector with 8 values
ID 5 → vector with 8 values
ID 7 → vector with 8 values
```

For an input:

```text
[2, 5, 7]
```

the output shape is:

```text
(1, 3, 8)
```

where:

```text
1 = batch size
3 = sequence length
8 = embedding dimension
```

---

# 6. Position Embedding

Token embeddings alone do not explicitly represent token position.

Position embedding is created with:

```python
self.position_embedding = nn.Embedding(
    max_sequence_length,
    embedding_dimension
)
```

For:

```text
max_sequence_length = 10
embedding_dimension = 8
```

the position embedding matrix has shape:

```text
(10, 8)
```

For a sequence of length 3:

```text
positions = [0, 1, 2]
```

the corresponding position vectors have shape:

```text
(1, 3, 8)
```

---

# 7. Token + Position

Token and position information are combined:

```python
x = token_vectors + position_vectors
```

Both tensors have the same shape:

```text
(1, 3, 8)
```

Therefore they can be added element-wise.

Conceptually:

```text
Token Information
        +
Position Information
        ↓
Input Representation
```

The resulting representation tells the model both:

- what the token represents
- where the token is located

---

# 8. Transformer Block

The Transformer Block contains:

```text
Self-Attention
      ↓
Residual + LayerNorm
      ↓
Feed Forward Network
      ↓
Residual + LayerNorm
```

The PyTorch implementation uses:

```python
nn.MultiheadAttention
```

for self-attention.

---

# 9. Self-Attention

The model uses:

```python
attention_output, _ = self.attention(
    x,
    x,
    x
)
```

The three arguments represent:

```text
query
key
value
```

Because all three are `x`:

```text
Q = x
K = x
V = x
```

This is self-attention.

In the earlier NumPy implementation, this process was written explicitly:

```text
Q = XWQ
K = XWK
V = XWV

Scores = QKᵀ
Weights = Softmax(Scores / √dk)

Output = Weights V
```

PyTorch's `nn.MultiheadAttention` handles these operations internally.

---

# 10. Multi-Head Attention

Example:

```text
embedding_dimension = 8
num_heads = 2
```

Each head operates on:

```text
8 / 2 = 4 dimensions
```

Conceptually:

```text
                8 dimensions
                     │
             ┌───────┴───────┐
             ↓               ↓
          Head 1           Head 2
          4 dims           4 dims
             └───────┬───────┘
                     ↓
                 Combined
                  8 dims
```

Multiple heads allow the model to learn different attention patterns.

The patterns are not manually assigned.

They are learned during training.

---

# 11. First Residual Connection

After attention:

```python
x = self.norm1(
    x + attention_output
)
```

Conceptually:

```text
X
│
├──────────────┐
│              │
↓              │
Attention      │
│              │
└────── + ─────┘
       ↓
   LayerNorm
```

Formula:

```text
X1 = LayerNorm(X + Attention(X))
```

The residual connection preserves a direct path for the original representation.

---

# 12. Feed Forward Network

The FFN is:

```python
self.ffn = nn.Sequential(
    nn.Linear(
        embedding_dimension,
        hidden_dimension
    ),
    nn.ReLU(),
    nn.Linear(
        hidden_dimension,
        embedding_dimension
    )
)
```

For:

```text
embedding_dimension = 8
hidden_dimension = 16
```

the transformation is:

```text
8 → 16 → 8
```

For a tensor:

```text
(1, 3, 8)
```

the intermediate representation becomes:

```text
(1, 3, 16)
```

and returns to:

```text
(1, 3, 8)
```

Formula:

```text
FFN(X) = ReLU(XW1 + b1)W2 + b2
```

---

# 13. Second Residual Connection

After the FFN:

```python
x = self.norm2(
    x + ffn_output
)
```

Formula:

```text
X2 = LayerNorm(X1 + FFN(X1))
```

Therefore the complete Transformer Block is:

```text
X
 ↓
Self-Attention
 ↓
Residual + LayerNorm
 ↓
FFN
 ↓
Residual + LayerNorm
 ↓
Output
```

---

# 14. Language Model Head

After the Transformer Block:

```python
self.lm_head = nn.Linear(
    embedding_dimension,
    vocabulary_size
)
```

Example:

```text
8 → 10
```

The input representation has 8 dimensions.

The vocabulary contains 10 possible tokens.

Therefore the output has 10 logits for every position.

Input:

```text
(1, 3, 8)
```

Output:

```text
(1, 3, 10)
```

Conceptually:

```text
Position 0
    ↓
10 logits

Position 1
    ↓
10 logits

Position 2
    ↓
10 logits
```

Each set of 10 logits corresponds to the 10 vocabulary tokens.

---

# 15. Logits

Logits are raw scores produced by the language model head.

Example:

```text
[1.2, -0.4, 2.7, 0.3, ...]
```

A higher logit means the model currently considers that token more likely relative to the others.

Logits are not probabilities.

Softmax can convert logits into probabilities:

```text
Logits
   ↓
Softmax
   ↓
Probabilities
```

---

# 16. Forward Pass

The complete forward pass is:

```text
Input IDs
    ↓
Token Embedding
    ↓
Position Embedding
    ↓
Token + Position
    ↓
Self-Attention
    ↓
Residual + LayerNorm
    ↓
FFN
    ↓
Residual + LayerNorm
    ↓
Language Model Head
    ↓
Logits
```

Example shape flow:

```text
Input IDs
(1, 3)
    ↓
Token Embedding
(1, 3, 8)
    +
Position Embedding
(1, 3, 8)
    ↓
Transformer Block
(1, 3, 8)
    ↓
Language Model Head
(1, 3, 10)
    ↓
Logits
(1, 3, 10)
```

---

# 17. What Does `(1, 3, 10)` Mean?

For:

```text
logits.shape = (1, 3, 10)
```

the dimensions mean:

```text
1  = batch size
3  = sequence length
10 = vocabulary size
```

So the model produces:

```text
Batch 0
│
├── Position 0 → 10 logits
├── Position 1 → 10 logits
└── Position 2 → 10 logits
```

---

# 18. Predicting the Next Token

To predict the next token, we use the logits from the last position:

```python
next_token_logits = logits[:, -1, :]
```

If:

```text
logits.shape = (1, 3, 10)
```

then:

```text
logits[:, -1, :]
```

has shape:

```text
(1, 10)
```

This gives the 10 logits for the last position.

Then greedy decoding can be used:

```python
next_token = torch.argmax(
    next_token_logits,
    dim=-1
)
```

The token with the highest logit is selected.

---

# 19. Autoregressive Generation

Generation works one token at a time.

Starting context:

```text
[2, 5, 7]
```

The model predicts:

```text
6
```

The new context becomes:

```text
[2, 5, 7, 6]
```

The model runs again:

```text
[2, 5, 7, 6]
        ↓
      predict
        ↓
        8
```

Then:

```text
[2, 5, 7, 6, 8]
```

This continues until the requested number of new tokens is generated.

Conceptually:

```text
Context
   ↓
Model
   ↓
Logits
   ↓
Token Selection
   ↓
Next Token
   ↓
Append
   ↓
New Context
   ↓
Model again
```

---

# 20. `torch.no_grad()`

Generation and experiments use:

```python
with torch.no_grad():
```

This tells PyTorch that gradients are not required for these operations.

During inference:

```text
No training
No backpropagation
No parameter updates
```

Therefore gradients do not need to be tracked.

This reduces memory usage and computation.

---

# 21. Parameter Count

PyTorch parameters can be inspected with:

```python
for name, parameter in model.named_parameters():
    print(name, parameter.shape)
```

The total number of parameters can be calculated with:

```python
total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)
```

`numel()` means:

```text
number of elements in the tensor
```

For example:

```text
shape = (10, 8)

10 × 8 = 80 parameters
```

---

# 22. Important Difference: Architecture vs Training

Stage 6 builds the architecture.

It does not train the model.

Current process:

```text
Random Initialization
        ↓
Forward Pass
        ↓
Random-ish Logits
        ↓
Token Selection
```

Therefore generated token IDs do not represent meaningful language.

Stage 7 will add:

```text
Training Data
      ↓
Forward Pass
      ↓
Logits
      ↓
Loss
      ↓
Backpropagation
      ↓
Gradients
      ↓
Parameter Update
      ↓
Repeat
```

After sufficient training, the parameters can learn useful patterns.

---

# 23. Stage 2 vs Stage 6

Stage 2 used NumPy:

```python
embedding = np.random.randn(...)
```

This demonstrated the mathematical concept of embeddings.

Stage 6 uses:

```python
nn.Embedding(...)
```

This creates a PyTorch learnable parameter.

The conceptual structure is still:

```text
Token ID
   ↓
Embedding Matrix
   ↓
Embedding Vector
```

The difference is that PyTorch integrates the embedding into a trainable neural network.

---

# 24. Stage 4 vs Stage 6

Stage 4 manually implemented Transformer concepts with NumPy.

Stage 6 uses PyTorch modules:

```text
Stage 4                    Stage 6

NumPy                      PyTorch
─────                      ───────

Manual Q/K/V         →     nn.MultiheadAttention

Manual FFN           →     nn.Linear

Manual LayerNorm     →     nn.LayerNorm

Manual parameters    →     nn.Parameter management

Manual pipeline      →     nn.Module
```

The underlying concepts remain the same.

Stage 6 is the transition from:

```text
Understanding the mathematics
```

to:

```text
Building a real neural network architecture
```

---

# 25. Complete Tiny LLM Mental Model

The most important mental model is:

```text
                    Tiny LLM
                       │
                       ▼
                  Token IDs
                       │
                       ▼
                Token Embedding
                       │
                       +
                Position Embedding
                       │
                       ▼
             Input Representation
                       │
                       ▼
          ┌─────────────────────────┐
          │    Transformer Block    │
          │                         │
          │  Self-Attention         │
          │       ↓                 │
          │  Residual + LayerNorm   │
          │       ↓                 │
          │  Feed Forward           │
          │       ↓                 │
          │  Residual + LayerNorm   │
          └────────────┬────────────┘
                       │
                       ▼
             Contextualized Vectors
                       │
                       ▼
              Language Model Head
                       │
                       ▼
                    Logits
                       │
                       ▼
                Token Selection
                       │
                       ▼
                  Next Token
                       │
                       ▼
               Append to Context
                       │
                       └──────→ Repeat
```

---

# 26. Key Concepts to Remember

### Token Embedding

```text
What is this token?
```

### Position Embedding

```text
Where is this token?
```

### Self-Attention

```text
Which other tokens should this token look at?
```

### FFN

```text
How should each token representation be transformed?
```

### Residual Connection

```text
Keep the original information and provide a shortcut.
```

### LayerNorm

```text
Stabilize the representation.
```

### Language Model Head

```text
Convert hidden representation into vocabulary scores.
```

### Logits

```text
Raw score for every vocabulary token.
```

### Generation

```text
Select one token → append → run model again.
```

---

# 27. Stage 6 Summary

The Tiny LLM now has the complete inference architecture:

```text
Token IDs
    ↓
Token Embedding
    ↓
Position Embedding
    ↓
Transformer Block
    ↓
Language Model Head
    ↓
Logits
    ↓
Next Token
```

However:

```text
Stage 6 = Build the model
Stage 7 = Train the model
```

The model currently has randomly initialized parameters.

The next major step is therefore to learn how these parameters are actually trained using:

```text
Dataset
 ↓
Input / Target
 ↓
Forward Pass
 ↓
Logits
 ↓
Loss
 ↓
Backpropagation
 ↓
Gradient
 ↓
Optimizer
 ↓
Parameter Update
 ↓
Repeat
```

That is the foundation of **LLM training**.