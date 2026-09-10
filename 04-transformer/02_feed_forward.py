import numpy as np

np.random.seed(42)

tokens = [
    "I",
    "love",
    "Japanese",
]

sequence_length = len(tokens)

embedding_dimension = 8
hidden_dimension = 16

X = np.random.randn(
    sequence_length,
    embedding_dimension
)

W1 = np.random.randn(
    embedding_dimension,
    hidden_dimension
)

b1 = np.zeros(hidden_dimension)

W2 = np.random.randn(
    hidden_dimension,
    embedding_dimension
)

b2 = np.zeros(embedding_dimension)


hidden = X @ W1 + b1

relu_output = np.maximum(0, hidden)

output = relu_output @ W2 + b2


print("Input:")
print(X)

print("\nInput shape:")
print(X.shape)

print("\nHidden layer:")
print(hidden)

print("\nHidden shape:")
print(hidden.shape)

print("\nAfter ReLU:")
print(relu_output)

print("\nOutput:")
print(output)

print("\nOutput shape:")
print(output.shape)