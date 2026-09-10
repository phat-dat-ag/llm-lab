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

transformation = np.random.randn(
    sequence_length,
    embedding_dimension
)

output = X + transformation


print("Input:")
print(X)

print("\nInput shape:")
print(X.shape)

print("\nTransformation:")
print(transformation)

print("\nTransformation shape:")
print(transformation.shape)

print("\nOutput:")
print(output)

print("\nOutput shape:")
print(output.shape)