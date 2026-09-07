import numpy as np

vocabulary_size = 10
embedding_dimension = 8

embedding_matrix = np.random.randn(
    vocabulary_size,
    embedding_dimension
)

print("Vocabulary size:")
print(vocabulary_size)

print("\nEmbedding dimension:")
print(embedding_dimension)

print("\nEmbedding matrix:")
print(embedding_matrix)

print("\nMatrix shape:")
print(embedding_matrix.shape)