import numpy as np

np.random.seed(42)

tokens = [
    "I",
    "love",
    "Japanese",
]

sequence_length = len(tokens)
embedding_dimension = 8

num_heads = 2
head_dimension = embedding_dimension // num_heads

X = np.random.randn(
    sequence_length,
    embedding_dimension
)


def attention(X):
    W_Q = np.random.randn(
        embedding_dimension,
        embedding_dimension
    )

    W_K = np.random.randn(
        embedding_dimension,
        embedding_dimension
    )

    W_V = np.random.randn(
        embedding_dimension,
        embedding_dimension
    )

    Q = X @ W_Q
    K = X @ W_K
    V = X @ W_V

    scores = Q @ K.T

    scaled_scores = (
        scores / np.sqrt(embedding_dimension)
    )

    exp_scores = np.exp(
        scaled_scores
        - np.max(scaled_scores, axis=-1, keepdims=True)
    )

    weights = (
        exp_scores
        / np.sum(exp_scores, axis=-1, keepdims=True)
    )

    return weights @ V


heads = []

for head in range(num_heads):
    head_output = attention(X)
    heads.append(head_output[:, :head_dimension])

multi_head_output = np.concatenate(
    heads,
    axis=-1
)

print("Input shape:")
print(X.shape)

print("\nNumber of heads:")
print(num_heads)

print("\nHead dimension:")
print(head_dimension)

print("\nMulti-head output:")
print(multi_head_output)

print("\nOutput shape:")
print(multi_head_output.shape)