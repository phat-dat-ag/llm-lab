import numpy as np

np.random.seed(42)


# =========================================================
# Configuration
# =========================================================

vocabulary_size = 6
embedding_dimension = 8


# =========================================================
# Transformer output
# =========================================================

transformer_output = np.random.randn(
    embedding_dimension
)


# =========================================================
# Output projection
# =========================================================

W_output = np.random.randn(
    embedding_dimension,
    vocabulary_size
)

b_output = np.zeros(
    vocabulary_size
)


logits = (
    transformer_output @ W_output
) + b_output


# =========================================================
# Results
# =========================================================

print("Transformer output:")
print(transformer_output)

print("\nTransformer output shape:")
print(transformer_output.shape)

print("\nOutput projection shape:")
print(W_output.shape)

print("\nLogits:")
print(logits)

print("\nLogits shape:")
print(logits.shape)