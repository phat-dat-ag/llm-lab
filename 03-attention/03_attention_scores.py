import numpy as np


np.random.seed(42)

tokens = [
    "I",
    "love",
    "Japanese",
]

embedding_dimension = 4
attention_dimension = 4

X = np.random.randn(
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

Q = X @ W_Q
K = X @ W_K


scores = Q @ K.T


print("Tokens:")
print(tokens)

print("\nAttention scores:")
print(scores)

print("\nScore shape:")
print(scores.shape)