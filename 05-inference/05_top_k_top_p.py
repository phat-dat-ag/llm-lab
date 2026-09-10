import numpy as np


# =========================================================
# Vocabulary and probabilities
# =========================================================

vocabulary = [
    "I",
    "love",
    "Japanese",
    "Python",
    "language",
    "."
]

probabilities = np.array([
    0.02,
    0.16,
    0.02,
    0.40,
    0.25,
    0.15
])


# =========================================================
# Top-K
# =========================================================

def top_k_sampling(probabilities, k):

    top_k_indices = np.argsort(
        probabilities
    )[-k:]

    filtered_probabilities = np.zeros_like(
        probabilities
    )

    filtered_probabilities[top_k_indices] = (
        probabilities[top_k_indices]
    )

    filtered_probabilities /= np.sum(
        filtered_probabilities
    )

    return filtered_probabilities


# =========================================================
# Top-P
# =========================================================

def top_p_sampling(probabilities, p):

    sorted_indices = np.argsort(
        probabilities
    )[::-1]

    sorted_probabilities = probabilities[
        sorted_indices
    ]

    cumulative_probabilities = np.cumsum(
        sorted_probabilities
    )

    mask = cumulative_probabilities <= p

    # Always keep at least the highest-probability token
    mask[0] = True

    selected_indices = sorted_indices[mask]

    filtered_probabilities = np.zeros_like(
        probabilities
    )

    filtered_probabilities[selected_indices] = (
        probabilities[selected_indices]
    )

    filtered_probabilities /= np.sum(
        filtered_probabilities
    )

    return filtered_probabilities


# =========================================================
# Results
# =========================================================

print("Original probabilities:")
print(probabilities)


print("\nTop-K (k=3):")

top_k_probabilities = top_k_sampling(
    probabilities,
    k=3
)

print(top_k_probabilities)


print("\nTop-P (p=0.80):")

top_p_probabilities = top_p_sampling(
    probabilities,
    p=0.80
)

print(top_p_probabilities)