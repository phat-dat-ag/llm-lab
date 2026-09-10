import torch
import torch.nn as nn


# =========================================================
# Configuration
# =========================================================

vocabulary_size = 10
embedding_dimension = 8


# =========================================================
# Tiny LLM
# =========================================================

class TinyLLM(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embedding_dimension
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension
        )


# =========================================================
# Create Model
# =========================================================

model = TinyLLM(
    vocabulary_size=vocabulary_size,
    embedding_dimension=embedding_dimension
)


# =========================================================
# Inspect Model
# =========================================================

print("Model:")
print(model)

print("\nModel parameters:")

for name, parameter in model.named_parameters():

    print(
        name,
        parameter.shape
    )