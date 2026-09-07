import numpy as np

# Each token is represented by an integer ID.
token_id = 5

# Embedding dimension.
embedding_dimension = 8

# Create one embedding vector for the token.
embedding_vector = np.random.randn(embedding_dimension)

print("Token ID:")
print(token_id)

print("\nEmbedding dimension:")
print(embedding_dimension)

print("\nEmbedding vector:")
print(embedding_vector)

print("\nVector shape:")
print(embedding_vector.shape)