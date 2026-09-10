import numpy as np


tokens = [
    "I",
    "love",
    "Japanese",
]

attention_weights = np.array([
    [0.2, 0.7, 0.1],
    [0.1, 0.8, 0.1],
    [0.1, 0.2, 0.7],
])


print("Tokens:")
print(tokens)

print("\nAttention matrix:")
print(attention_weights)


for i, token in enumerate(tokens):

    print(f"\n{token} attends to:")

    for j, target_token in enumerate(tokens):

        weight = attention_weights[i][j]

        print(
            f"{target_token}: "
            f"{weight:.2f}"
        )