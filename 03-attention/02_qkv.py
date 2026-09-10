import numpy as np


np.random.seed(42)

tokens = [
    "I",
    "love",
    "Japanese",
]

embedding_dimension = 4
attention_dimension = 4

embedding_matrix = np.random.randn(
    len(tokens),
    embedding_dimension
)

W_Q = np.random.randn(
    embedding_dimension,
    attention_dimension
)

W_K = np.random.randn(
    embedding_dimension,
    attention_dimension
)

W_V = np.random.randn(
    embedding_dimension,
    attention_dimension
)


Q = embedding_matrix @ W_Q
K = embedding_matrix @ W_K
V = embedding_matrix @ W_V


print("Embedding matrix:")
print(embedding_matrix)

print("\nQ (Query):")
print(Q)

print("\nK (Key):")
print(K)

print("\nV (Value):")
print(V)