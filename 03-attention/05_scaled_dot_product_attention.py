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

W_V = np.random.randn(
    embedding_dimension,
    attention_dimension
)


Q = X @ W_Q
K = X @ W_K
V = X @ W_V


scores = Q @ K.T

scaled_scores = scores / np.sqrt(attention_dimension)


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))

    return exp_x / np.sum(
        exp_x,
        axis=-1,
        keepdims=True
    )


attention_weights = softmax(scaled_scores)

output = attention_weights @ V


print("Tokens:")
print(tokens)

print("\nQ:")
print(Q)

print("\nK:")
print(K)

print("\nV:")
print(V)

print("\nRaw attention scores:")
print(scores)

print("\nScaled attention scores:")
print(scaled_scores)

print("\nAttention weights:")
print(attention_weights)

print("\nAttention output:")
print(output)