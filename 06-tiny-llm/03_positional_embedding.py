import torch
import torch.nn as nn


# =========================================================
# Configuration
# =========================================================

vocabulary_size = 10
embedding_dimension = 8
max_sequence_length = 5


# =========================================================
# Token Embedding
# =========================================================

token_embedding = nn.Embedding(
    vocabulary_size,
    embedding_dimension
)


# =========================================================
# Positional Embedding
# =========================================================

position_embedding = nn.Embedding(
    max_sequence_length,
    embedding_dimension
)


# =========================================================
# Input Token IDs
# =========================================================

input_ids = torch.tensor([
    [2, 5, 7]
])


# =========================================================
# Sequence Length
# =========================================================

sequence_length = input_ids.shape[1]


# =========================================================
# Token Positions
# =========================================================

positions = torch.arange(
    sequence_length
)

positions = positions.unsqueeze(0)


# =========================================================
# Get Embeddings
# =========================================================

token_vectors = token_embedding(
    input_ids
)

position_vectors = position_embedding(
    positions
)


# =========================================================
# Combine Token + Position
# =========================================================

x = token_vectors + position_vectors


# =========================================================
# Results
# =========================================================

print("Input token IDs:")
print(input_ids)

print("\nInput shape:")
print(input_ids.shape)

print("\nPositions:")
print(positions)

print("\nPositions shape:")
print(positions.shape)

print("\nToken embedding shape:")
print(token_vectors.shape)

print("\nPosition embedding shape:")
print(position_vectors.shape)

print("\nFinal representation:")
print(x)

print("\nFinal representation shape:")
print(x.shape)