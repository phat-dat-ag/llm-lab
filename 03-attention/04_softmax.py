import numpy as np


def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)


scores = np.array([
    2.0,
    4.0,
    1.0
])

weights = softmax(scores)

print("Scores:")
print(scores)

print("\nAttention weights:")
print(weights)

print("\nSum:")
print(np.sum(weights))