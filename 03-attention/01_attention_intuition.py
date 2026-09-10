import numpy as np


tokens = [
    "I",
    "love",
    "Japanese",
]

embedding_dimension = 4

np.random.seed(42)

embedding_matrix = np.random.randn(
    len(tokens),
    embedding_dimension
)

print("Tokens:")
print(tokens)

print("\nEmbedding vectors:")

for i, token in enumerate(tokens):
    print(f"\n{token}:")
    print(embedding_matrix[i])