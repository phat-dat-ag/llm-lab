import torch
import torch.nn as nn


# =========================================================
# Configuration
# =========================================================

vocabulary_size = 10
embedding_dimension = 8


# =========================================================
# Language Model Head
# =========================================================

lm_head = nn.Linear(
    embedding_dimension,
    vocabulary_size
)


# =========================================================
# Simulated Transformer Output
# =========================================================

transformer_output = torch.randn(
    1,
    3,
    embedding_dimension
)


# =========================================================
# Project to Vocabulary
# =========================================================

logits = lm_head(
    transformer_output
)


# =========================================================
# Results
# =========================================================

print("Transformer output:")
print(transformer_output)

print("\nTransformer output shape:")
print(transformer_output.shape)

print("\nLanguage Model Head:")
print(lm_head)

print("\nLanguage Model Head weight shape:")
print(lm_head.weight.shape)

print("\nLanguage Model Head bias shape:")
print(lm_head.bias.shape)

print("\nLogits:")
print(logits)

print("\nLogits shape:")
print(logits.shape)