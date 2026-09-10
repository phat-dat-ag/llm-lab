import numpy as np

np.random.seed(42)


# =========================================================
# Configuration
# =========================================================

vocabulary = [
    "I",
    "love",
    "Japanese",
    "Python",
    "language",
    "."
]

vocabulary_size = len(vocabulary)

max_new_tokens = 5


# =========================================================
# Initial context
# =========================================================

context = [
    "I",
    "love"
]


# =========================================================
# Softmax
# =========================================================

def softmax(logits):

    exp_logits = np.exp(
        logits - np.max(logits)
    )

    return exp_logits / np.sum(exp_logits)


# =========================================================
# Simulated model
# =========================================================

def model(context):

    """
    Simulate model logits based on current context.
    This is NOT a trained language model.
    """

    logits = np.random.randn(
        vocabulary_size
    )

    return logits


# =========================================================
# Autoregressive generation
# =========================================================

print("Initial context:")
print(context)


for step in range(max_new_tokens):

    print(f"\n--- Step {step + 1} ---")

    # -----------------------------------------
    # 1. Model receives current context
    # -----------------------------------------

    logits = model(context)

    print("Context:")
    print(context)

    print("\nLogits:")
    print(logits)

    # -----------------------------------------
    # 2. Convert logits to probabilities
    # -----------------------------------------

    probabilities = softmax(logits)

    print("\nProbabilities:")
    print(probabilities)

    # -----------------------------------------
    # 3. Select next token
    # -----------------------------------------

    next_token_id = np.argmax(
        probabilities
    )

    next_token = vocabulary[
        next_token_id
    ]

    print("\nSelected token:")
    print(next_token)

    # -----------------------------------------
    # 4. Append token to context
    # -----------------------------------------

    context.append(
        next_token
    )

    print("\nUpdated context:")
    print(context)


# =========================================================
# Final result
# =========================================================

print("\n==============================")
print("Final generated text:")
print(" ".join(context))