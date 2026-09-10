import numpy as np

np.random.seed(42)

sequence_length = 3
embedding_dimension = 8
hidden_dimension = 16


# =========================================================
# Transformer Block
# =========================================================

def transformer_block(X):

    # -------------------------
    # 1. Self-Attention
    # -------------------------

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

    attention_weights = (
        exp_scores
        / np.sum(
            exp_scores,
            axis=-1,
            keepdims=True
        )
    )

    attention_output = (
        attention_weights @ V
    )

    # -------------------------
    # 2. Residual + LayerNorm
    # -------------------------

    residual_1 = X + attention_output

    mean_1 = np.mean(
        residual_1,
        axis=-1,
        keepdims=True
    )

    variance_1 = np.var(
        residual_1,
        axis=-1,
        keepdims=True
    )

    normalized_1 = (
        residual_1 - mean_1
    ) / np.sqrt(
        variance_1 + 1e-5
    )

    # -------------------------
    # 3. Feed Forward Network
    # -------------------------

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

    hidden = normalized_1 @ W1 + b1

    relu_output = np.maximum(
        0,
        hidden
    )

    ffn_output = (
        relu_output @ W2
    ) + b2

    # -------------------------
    # 4. Residual + LayerNorm
    # -------------------------

    residual_2 = (
        normalized_1 + ffn_output
    )

    mean_2 = np.mean(
        residual_2,
        axis=-1,
        keepdims=True
    )

    variance_2 = np.var(
        residual_2,
        axis=-1,
        keepdims=True
    )

    output = (
        residual_2 - mean_2
    ) / np.sqrt(
        variance_2 + 1e-5
    )

    return (
        attention_weights,
        attention_output,
        normalized_1,
        ffn_output,
        output
    )


# =========================================================
# Experiment
# =========================================================

X = np.random.randn(
    sequence_length,
    embedding_dimension
)

(
    attention_weights,
    attention_output,
    normalized_1,
    ffn_output,
    output
) = transformer_block(X)


# =========================================================
# Results
# =========================================================

print("Input:")
print(X)

print("\nInput shape:")
print(X.shape)

print("\nAttention weights:")
print(attention_weights)

print("\nAttention output:")
print(attention_output)

print("\nFirst LayerNorm output:")
print(normalized_1)

print("\nFFN output:")
print(ffn_output)

print("\nFinal Transformer output:")
print(output)

print("\nFinal output shape:")
print(output.shape)