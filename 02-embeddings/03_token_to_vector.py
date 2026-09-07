import numpy as np

vocabulary_size = 10
embedding_dimension = 8

embedding_matrix = np.random.randn(
    vocabulary_size,
    embedding_dimension
)

token_ids = [2, 5, 7]

print("Token IDs:")
print(token_ids)

print("\nEmbedding vectors:")

for token_id in token_ids:
    vector = embedding_matrix[token_id]

    print(f"\nToken ID: {token_id}")
    print(vector)