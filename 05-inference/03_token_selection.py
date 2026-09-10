import numpy as np

np.random.seed(42)


# =========================================================
# Vocabulary
# =========================================================

vocabulary = [
    "I",
    "love",
    "Japanese",
    "Python",
    "language",
    "."
]


# =========================================================
# Probabilities
# =========================================================

probabilities = np.array([
    0.020,
    0.165,
    0.020,
    0.400,
    0.250,
    0.145
])


# =========================================================
# 1. Greedy Selection
# =========================================================

greedy_index = np.argmax(probabilities)

greedy_token = vocabulary[greedy_index]


# =========================================================
# 2. Random Sampling
# =========================================================

sampled_index = np.random.choice(
    len(vocabulary),
    p=probabilities
)

sampled_token = vocabulary[sampled_index]


# =========================================================
# Results
# =========================================================

print("Vocabulary:")
print(vocabulary)

print("\nProbabilities:")
print(probabilities)

print("\nGreedy selection:")
print("Index:", greedy_index)
print("Token:", greedy_token)

print("\nRandom sampling:")
print("Index:", sampled_index)
print("Token:", sampled_token)