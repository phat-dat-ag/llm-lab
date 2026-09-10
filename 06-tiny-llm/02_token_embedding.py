import torch
import torch.nn as nn


# =========================================================
# Configuration
# =========================================================

vocabulary_size = 10
embedding_dimension = 8


# =========================================================
# Token Embedding
# =========================================================

embedding = nn.Embedding(
    vocabulary_size,
    embedding_dimension
)


# =========================================================
# Input Token IDs
# =========================================================

input_ids = torch.tensor([
    [2, 5, 7]
])


# =========================================================
# Convert Token IDs to Embeddings
# =========================================================

embedded = embedding(
    input_ids
)


# =========================================================
# Results
# =========================================================

print("Input token IDs:")
print(input_ids)

print("\nInput shape:")
print(input_ids.shape)

print("\nEmbedding:")
print(embedded)

print("\nEmbedding shape:")
print(embedded.shape)

print("\nEmbedding weight shape:")
print(embedding.weight.shape)