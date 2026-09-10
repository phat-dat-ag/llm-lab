import numpy as np

np.random.seed(42)

sequence_length = 3
embedding_dimension = 8
hidden_dimension = 16

X = np.random.randn(
    sequence_length,
    embedding_dimension
)


# =========================================================
# 1. Multi-Head Self-Attention
# =========================================================

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
        - np.max(
            scaled_scores,
            axis=-1,
            keepdims=True
        )
    )

    weights = (
        exp_scores
        / np.sum(
            exp_scores,
            axis=-1,
            keepdims=True
        )
    )

    return weights @ V


attention_output = attention(X)


# =========================================================
# 2. Residual Connection
# =========================================================

residual_1 = X + attention_output


# =========================================================
# 3. Layer Normalization
# =========================================================

def layer_norm(X):

    gamma = np.ones(
        embedding_dimension
    )

    beta = np.zeros(
        embedding_dimension
    )

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

    return gamma * normalized + beta


norm_1 = layer_norm(residual_1)


# =========================================================
# 4. Feed Forward Network
# =========================================================

W1 = np.random.randn(
    embedding_dimension,
    hidden_dimension
)

b1 = np.zeros(
    hidden_dimension
)

W2 = np.random.randn(
    hidden_dimension,
    embedding_dimension
)

b2 = np.zeros(
    embedding_dimension
)

hidden = norm_1 @ W1 + b1

relu_output = np.maximum(
    0,
    hidden
)

ffn_output = (
    relu_output @ W2
) + b2


# =========================================================
# 5. Residual Connection
# =========================================================

residual_2 = norm_1 + ffn_output


# =========================================================
# 6. Layer Normalization
# =========================================================

output = layer_norm(
    residual_2
)


# =========================================================
# Results
# =========================================================

print("Input shape:")
print(X.shape)

print("\nAfter Attention:")
print(attention_output.shape)

print("\nAfter Residual 1:")
print(residual_1.shape)

print("\nAfter LayerNorm 1:")
print(norm_1.shape)

print("\nAfter FFN:")
print(ffn_output.shape)

print("\nAfter Residual 2:")
print(residual_2.shape)

print("\nFinal Output:")
print(output)

print("\nFinal Output shape:")
print(output.shape)