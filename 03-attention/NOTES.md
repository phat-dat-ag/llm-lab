# Stage 3 — Attention

## Goal

Understand how a token can use information from other tokens.

## Core pipeline

Embedding
→ Q, K, V
→ QKᵀ
→ Scaling
→ Softmax
→ Attention Weights
→ Weighted Sum of V

## Formula

Attention(Q, K, V)
=
softmax(QKᵀ / √dₖ)V

## Key concepts

- Query
- Key
- Value
- Attention Score
- Scaling
- Softmax
- Attention Weight
- Contextualized Representation