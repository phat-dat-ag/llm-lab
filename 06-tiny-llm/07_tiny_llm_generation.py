import torch
import torch.nn as nn


# =========================================================
# Configuration
# =========================================================

vocabulary_size = 10
embedding_dimension = 8
hidden_dimension = 16
max_sequence_length = 10
num_heads = 2


# =========================================================
# Tiny LLM
# =========================================================

class TinyLLM(nn.Module):

    def __init__(
        self,
        vocabulary_size,
        embedding_dimension,
        hidden_dimension,
        max_sequence_length,
        num_heads
    ):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension
        )

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dimension
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=num_heads,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(
            embedding_dimension
        )

        self.ffn = nn.Sequential(
            nn.Linear(
                embedding_dimension,
                hidden_dimension
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_dimension,
                embedding_dimension
            )
        )

        self.norm2 = nn.LayerNorm(
            embedding_dimension
        )

        self.lm_head = nn.Linear(
            embedding_dimension,
            vocabulary_size
        )


    def forward(self, input_ids):

        # ---------------------------------------------
        # Token Embedding
        # ---------------------------------------------

        token_vectors = self.token_embedding(
            input_ids
        )

        # ---------------------------------------------
        # Position Embedding
        # ---------------------------------------------

        sequence_length = input_ids.shape[1]

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        ).unsqueeze(0)

        position_vectors = self.position_embedding(
            positions
        )

        # ---------------------------------------------
        # Token + Position
        # ---------------------------------------------

        x = token_vectors + position_vectors

        # ---------------------------------------------
        # Self-Attention
        # ---------------------------------------------

        attention_output, _ = self.attention(
            x,
            x,
            x
        )

        # ---------------------------------------------
        # Residual + LayerNorm
        # ---------------------------------------------

        x = self.norm1(
            x + attention_output
        )

        # ---------------------------------------------
        # Feed Forward
        # ---------------------------------------------

        ffn_output = self.ffn(x)

        # ---------------------------------------------
        # Residual + LayerNorm
        # ---------------------------------------------

        x = self.norm2(
            x + ffn_output
        )

        # ---------------------------------------------
        # Language Model Head
        # ---------------------------------------------

        logits = self.lm_head(
            x
        )

        return logits


# =========================================================
# Create Model
# =========================================================

model = TinyLLM(
    vocabulary_size=vocabulary_size,
    embedding_dimension=embedding_dimension,
    hidden_dimension=hidden_dimension,
    max_sequence_length=max_sequence_length,
    num_heads=num_heads
)


# =========================================================
# Initial Context
# =========================================================

input_ids = torch.tensor([
    [2, 5, 7]
])


# =========================================================
# Generation
# =========================================================

num_new_tokens = 5

generated_ids = input_ids.clone()


with torch.no_grad():

    for step in range(num_new_tokens):

        logits = model(
            generated_ids
        )

        # Get logits from the last position
        next_token_logits = logits[:, -1, :]

        # Greedy token selection
        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        # Append new token
        generated_ids = torch.cat(
            [
                generated_ids,
                next_token
            ],
            dim=1
        )


# =========================================================
# Results
# =========================================================

print("Initial input IDs:")
print(input_ids)

print("\nInitial shape:")
print(input_ids.shape)

print("\nGenerated IDs:")
print(generated_ids)

print("\nGenerated shape:")
print(generated_ids.shape)

print("\nNumber of new tokens:")
print(num_new_tokens)