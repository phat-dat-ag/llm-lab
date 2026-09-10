import numpy as np


# =========================================================
# Logits
# =========================================================

logits = np.array([
    1.0,
    2.0,
    3.0,
    4.0
])


# =========================================================
# Temperature Softmax
# =========================================================

def softmax_with_temperature(logits, temperature):

    scaled_logits = logits / temperature

    exp_logits = np.exp(
        scaled_logits - np.max(scaled_logits)
    )

    probabilities = (
        exp_logits
        / np.sum(exp_logits)
    )

    return probabilities


# =========================================================
# Different temperatures
# =========================================================

temperatures = [
    0.5,
    1.0,
    2.0
]


for temperature in temperatures:

    probabilities = softmax_with_temperature(
        logits,
        temperature
    )

    print(f"\nTemperature: {temperature}")

    print("Probabilities:")
    print(probabilities)

    print("Sum:")
    print(np.sum(probabilities))