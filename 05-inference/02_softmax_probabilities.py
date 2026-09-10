import numpy as np

np.random.seed(42)


# =========================================================
# Logits
# =========================================================

logits = np.array([
    -2.57046341,
    -0.45753050,
    -2.57364524,
     0.74274850,
     0.28876848,
     0.59768005
])


# =========================================================
# Softmax
# =========================================================

def softmax(x):

    exp_x = np.exp(
        x - np.max(x)
    )

    return exp_x / np.sum(exp_x)


probabilities = softmax(logits)


# =========================================================
# Results
# =========================================================

print("Logits:")
print(logits)

print("\nProbabilities:")
print(probabilities)

print("\nProbability sum:")
print(np.sum(probabilities))

print("\nHighest probability index:")
print(np.argmax(probabilities))

print("\nHighest probability:")
print(np.max(probabilities))