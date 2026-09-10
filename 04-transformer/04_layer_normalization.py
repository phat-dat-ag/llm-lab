import numpy as np

np.random.seed(42)

tokens = [
    "I",
    "love",
    "Japanese",
]

sequence_length = len(tokens)
embedding_dimension = 8

X = np.random.randn(
    sequence_length,
    embedding_dimension
)

gamma = np.ones(embedding_dimension)
beta = np.zeros(embedding_dimension)

mean = np.mean(
    X,
    axis=-1,
    keepdims=True
)

variance = np.var(
    X,
    axis=-1,
    keepdims=True
)

epsilon = 1e-5

normalized = (
    X - mean
) / np.sqrt(
    variance + epsilon
)

output = gamma * normalized + beta


print("Input:")
print(X)

print("\nMean:")
print(mean)

print("\nVariance:")
print(variance)

print("\nNormalized:")
print(normalized)

print("\nOutput:")
print(output)

print("\nOutput shape:")
print(output.shape)