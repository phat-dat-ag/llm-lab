# Inference

## 1. Overview

Inference is the process of using a trained language model to generate text.

The basic pipeline is:

```text
Input Text
    ↓
Tokenization
    ↓
Token IDs
    ↓
Embedding
    ↓
Transformer
    ↓
Hidden Representation
    ↓
Output Projection
    ↓
Logits
    ↓
Softmax
    ↓
Probabilities
    ↓
Token Selection
    ↓
Next Token
    ↓
Append to Context
    ↓
Repeat
```

During inference, the model does **not** update its parameters.

Training changes model parameters.

Inference uses the learned parameters to predict the next token.

---

## 2. Logits

The Transformer produces a hidden representation for the current context.

For next-token prediction, this representation is projected into vocabulary space.

```text
Transformer Output
        ↓
Output Projection
        ↓
Logits
```

Suppose:

```text
hidden_dimension = 8
vocabulary_size = 6
```

Then:

```text
Transformer output: (8,)
Output projection:  (8, 6)
Logits:             (6,)
```

The output projection is:

```python
logits = hidden @ W_output + b_output
```

where:

```text
hidden      → Transformer representation
W_output    → output projection weights
b_output    → output bias
logits      → one score for every vocabulary token
```

Therefore:

```text
1 logit ↔ 1 vocabulary token
```

Example:

```text
ID    Token       Logit
0     I          -2.57
1     love       -0.46
2     Japanese   -2.57
3     Python      0.74
4     language    0.29
5     .           0.60
```

The highest logit is `Python`.

However, logits are **not probabilities**.

A logit of `0.74` does NOT mean `74%`.

Logits are raw scores that will later be converted into probabilities.

---

## 3. Softmax

Softmax converts logits into a probability distribution.

Formula:

```text
P_i = exp(x_i) / Σ exp(x_j)
```

In code:

```python
def softmax(x):

    exp_x = np.exp(
        x - np.max(x)
    )

    return exp_x / np.sum(exp_x)
```

The subtraction:

```python
x - np.max(x)
```

is used for numerical stability.

It does not change the resulting Softmax probabilities.

After Softmax:

```text
0 ≤ P_i ≤ 1
```

and:

```text
Σ P_i = 1
```

Example:

```text
Token       Probability
Python        43.29%
Japanese      18.04%
I             15.51%
love           8.22%
language       7.47%
.              7.47%
```

Now the model has a probability distribution over the vocabulary.

The process is:

```text
Logits
  ↓
Softmax
  ↓
Probabilities
```

---

## 4. Token Selection

After obtaining probabilities, the model must select the next token.

There are several decoding strategies.

### 4.1 Greedy Decoding

Greedy decoding selects the token with the highest probability.

```python
next_token_id = np.argmax(probabilities)
```

Example:

```text
Python      40%
language    25%
love        16%
.           15%
I             2%
Japanese      2%
```

Greedy decoding always selects:

```text
Python
```

Advantages:

- Simple
- Deterministic
- Predictable

Disadvantages:

- Can be repetitive
- Can produce less diverse text

---

### 4.2 Random Sampling

Sampling selects a token according to the probability distribution.

```python
next_token_id = np.random.choice(
    len(vocabulary),
    p=probabilities
)
```

For example:

```text
Python      40%
language    25%
love        16%
.           15%
I             2%
Japanese      2%
```

Python is the most likely choice, but other tokens can also be selected.

Therefore:

```text
Greedy   → always choose highest probability
Sampling → choose according to probabilities
```

Sampling introduces randomness and diversity.

---

## 5. Temperature

Temperature modifies the logits before Softmax.

Formula:

```text
P_i = exp(x_i / T) / Σ exp(x_j / T)
```

where:

```text
T = temperature
```

In code:

```python
scaled_logits = logits / temperature
```

Then:

```text
Scaled Logits
      ↓
    Softmax
      ↓
 Probabilities
```

### 5.1 Low Temperature

When:

```text
T < 1
```

the probability distribution becomes sharper.

The model strongly favors high-probability tokens.

```text
Low temperature
      ↓
More deterministic
      ↓
Less diversity
```

### 5.2 Temperature = 1

When:

```text
T = 1
```

the normal Softmax distribution is used.

### 5.3 High Temperature

When:

```text
T > 1
```

the probability distribution becomes flatter.

Lower-probability tokens become more likely to be selected.

```text
High temperature
      ↓
More randomness
      ↓
More diversity
```

Important:

> Temperature changes the probability distribution. It does not directly select the token.

Also:

> Lower temperature does not mean the model is inherently smarter.

It only changes the decoding behavior.

---

## 6. Top-K Sampling

Top-K limits sampling to the K highest-probability tokens.

Example:

```text
Python      40%
language    25%
love        16%
.           15%
I             2%
Japanese      2%
```

With:

```text
K = 3
```

keep:

```text
Python
language
love
```

and remove:

```text
.
I
Japanese
```

The remaining probabilities are then normalized again.

Conceptually:

```text
All vocabulary
      ↓
Keep K highest probabilities
      ↓
Renormalize
      ↓
Sample
```

Top-K always keeps exactly K candidates.

---

## 7. Top-P / Nucleus Sampling

Top-P selects a variable number of high-probability tokens based on cumulative probability.

First, sort tokens by probability:

```text
Python      40%
language    25%
love        16%
.           15%
I             2%
Japanese      2%
```

For:

```text
P = 0.80
```

the cumulative probability becomes:

```text
Python                         0.40
Python + language              0.65
Python + language + love       0.81
```

Therefore the candidate set becomes:

```text
Python
language
love
```

The selected tokens are then renormalized.

Conceptually:

```text
All vocabulary
      ↓
Sort by probability
      ↓
Take smallest prefix reaching P
      ↓
Renormalize
      ↓
Sample
```

Unlike Top-K:

```text
Top-K → fixed number of candidates
Top-P → variable number of candidates
```

For example:

```text
Top-K:
K = 3
→ always 3 candidates

Top-P:
P = 0.80
→ number of candidates depends on distribution
```

If the probability distribution is very concentrated, Top-P may keep only a few tokens.

If the probability distribution is very spread out, it may keep many tokens.

### Important implementation detail

In the educational implementation used in this lab, Top-P was implemented approximately as:

```python
mask = cumulative_probabilities <= p
mask[0] = True
```

A production-style Top-P implementation normally keeps the **smallest prefix whose cumulative probability reaches or exceeds `p`**.

For example, if:

```text
0.40 + 0.25 + 0.16 = 0.81
```

and:

```text
p = 0.80
```

the third token is kept because it is the token that crosses the threshold.

The important concept is:

```text
Top-P
  ↓
Sort by probability
  ↓
Accumulate probability
  ↓
Stop when cumulative probability reaches P
  ↓
Sample from remaining candidates
```

---

## 8. Autoregressive Generation

An autoregressive language model generates text one token at a time.

Example:

```text
Initial context:

I love
```

The model predicts the next token:

```text
I love → Python
```

The new token is appended to the context:

```text
I love Python
```

The model then predicts another token:

```text
I love Python → I
```

The process continues.

```text
Current Context
       ↓
     Model
       ↓
     Logits
       ↓
    Softmax
       ↓
 Probabilities
       ↓
 Token Selection
       ↓
   Next Token
       ↓
Append to Context
       ↓
     Repeat
```

This is called **autoregressive generation** because each new token is generated based on the previous context.

---

## 9. Autoregressive Probability

For a sequence:

```text
x₁, x₂, x₃, ..., xₙ
```

the probability can be expressed as:

```text
P(x₁, x₂, ..., xₙ)
=
P(x₁)
× P(x₂ | x₁)
× P(x₃ | x₁, x₂)
× ...
× P(xₙ | x₁, ..., xₙ₋₁)
```

The important idea is:

```text
Next token
    depends on
Previous context
```

Therefore generation happens sequentially.

---

## 10. Context Window vs max_new_tokens

These are different concepts.

### 10.1 Context Window

The context window is the maximum amount of token context the model can process within its supported limit.

Conceptually:

```text
Prompt
  +
Conversation / history
  +
Generated tokens
```

must fit within the model's usable context limit.

### 10.2 max_new_tokens

`max_new_tokens` specifies how many new tokens to generate.

Example:

```python
max_new_tokens = 5
```

means:

```text
Generate at most 5 new tokens
```

It does **not** mean:

```text
Context window = 5
```

Example:

```text
Initial context = 2 tokens

max_new_tokens = 5

Final sequence = 2 + 5 = 7 tokens
```

---

## 11. Toy Model vs Real LLM

The `06_autoregressive_generation.py` experiment intentionally uses:

```python
def model(context):

    logits = np.random.randn(
        vocabulary_size
    )

    return logits
```

This is **not a trained language model**.

The generated logits are random.

Therefore the model does not actually understand:

```text
"I love"
```

or:

```text
"I love Python"
```

The output can therefore look like:

```text
I love Python I . Japanese Python
```

This is expected.

The purpose of the experiment is to demonstrate the **generation loop**, not language understanding.

In a real LLM:

```text
Context
   ↓
Embedding
   ↓
Transformer
   ↓
Hidden Representation
   ↓
Output Projection
   ↓
Logits
```

The Transformer uses the context to produce meaningful representations.

Therefore the logits depend on the input context.

---

## 12. Full Inference Pipeline

Putting everything together:

```text
                Prompt
                  ↓
             Tokenization
                  ↓
              Token IDs
                  ↓
              Embedding
                  ↓
             Transformer
                  ↓
        Hidden Representation
                  ↓
         Output Projection
                  ↓
                Logits
                  ↓
               Softmax
                  ↓
            Probabilities
                  ↓
          Decoding Strategy
        ┌─────────┼──────────┐
        ↓         ↓          ↓
     Greedy   Temperature  Top-K/Top-P
        └─────────┼──────────┘
                  ↓
             Next Token
                  ↓
          Append to Context
                  ↓
               Repeat
```

The generation loop is:

```text
Context
   ↓
Transformer
   ↓
Logits
   ↓
Probabilities
   ↓
Select token
   ↓
Append token
   ↓
Context updated
   ↓
Transformer again
   ↓
...
```

---

## 13. Important Distinctions

### Token ID

An index into the vocabulary.

```text
token → token ID
```

A token ID does not contain semantic meaning by itself.

### Embedding

A learned vector representation of a token.

```text
token ID → embedding vector
```

### Transformer

Processes contextual relationships between tokens.

```text
embeddings
    ↓
Transformer
    ↓
contextual representations
```

### Logits

Raw scores for every possible next token.

```text
representation → logits
```

### Probabilities

Normalized likelihood distribution.

```text
logits → Softmax → probabilities
```

### Decoding

The process of choosing the next token from the probability distribution.

```text
probabilities → decoding → next token
```

---

## 14. Core Mental Model

The most important idea from this stage is:

```text
Transformer
    ↓
"What could the next token be?"
    ↓
Logits
    ↓
"How likely is each token?"
    ↓
Softmax
    ↓
"Which token should we choose?"
    ↓
Decoding
    ↓
Next Token
    ↓
"Add it to the context"
    ↓
Repeat
```

In one sentence:

> **Inference is the process of repeatedly predicting and selecting the next token using the model's learned parameters.**

---

## 15. Stage 5 Summary

The complete Stage 5 pipeline is:

```text
Logits
   ↓
Softmax
   ↓
Probabilities
   ↓
Decoding
   ├── Greedy
   ├── Sampling
   ├── Temperature
   ├── Top-K
   └── Top-P
   ↓
Next Token
   ↓
Append to Context
   ↓
Repeat
```

Key concepts:

- Logits are raw scores.
- Softmax converts logits into probabilities.
- Greedy selects the highest-probability token.
- Sampling introduces randomness.
- Temperature controls probability distribution sharpness.
- Top-K keeps a fixed number of candidates.
- Top-P keeps an adaptive probability mass.
- Autoregressive generation produces one token at a time.
- `max_new_tokens` controls generated length, not context-window size.
- The toy model demonstrates the inference mechanism but is not a trained LLM.
- Real LLM inference uses learned Transformer parameters to make context-dependent predictions.

---

## 16. Stage 5 Mental Model

At the end of Stage 5, the overall LLM inference process can be understood as:

```text
User Prompt
     ↓
Tokenization
     ↓
Token IDs
     ↓
Embedding
     ↓
Transformer
     ↓
Contextual Representation
     ↓
Output Projection
     ↓
Logits
     ↓
Softmax
     ↓
Probability Distribution
     ↓
Decoding
     ↓
Next Token
     ↓
Append to Context
     ↓
Run Transformer Again
     ↓
Predict Next Token
     ↓
...
```

The model repeats this process until:

- A stop condition is reached.
- An end-of-sequence token is generated.
- `max_new_tokens` is reached.
- The available context limit is reached.

The fundamental idea is:

```text
Predict
  ↓
Select
  ↓
Append
  ↓
Predict again
  ↓
Repeat
```

This is the core mechanism behind autoregressive text generation.