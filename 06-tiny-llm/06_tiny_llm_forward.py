import torch
import torch.nn as nn


# =========================================================
# Configuration
# =========================================================

vocabulary_size = 10
embedding_dimension = 8
hidden_dimension = 16
max_sequence_length = 5
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

        # ---------------------------------------------
        # 1. Token Embedding
        # ---------------------------------------------

        self.token_embedding = nn.Embedding(
            vocabulary_size,
            embedding_dimension
        )

        # ---------------------------------------------
        # 2. Position Embedding
        # ---------------------------------------------

        self.position_embedding = nn.Embedding(
            max_sequence_length,
            embedding_dimension
        )

        # ---------------------------------------------
        # 3. Transformer Block
        # ---------------------------------------------

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

        # ---------------------------------------------
        # 4. Language Model Head
        # ---------------------------------------------

        self.lm_head = nn.Linear(
            embedding_dimension,
            vocabulary_size
        )


    def forward(self, input_ids):

        # ---------------------------------------------
        # 1. Token Embedding
        # ---------------------------------------------

        token_vectors = self.token_embedding(
            input_ids
        )

        # ---------------------------------------------
        # 2. Position Embedding
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
        # 3. Combine Token + Position
        # ---------------------------------------------

        x = token_vectors + position_vectors

        # ---------------------------------------------
        # 4. Self-Attention
        # ---------------------------------------------

        attention_output, _ = self.attention(
            x,
            x,
            x
        )

        # ---------------------------------------------
        # 5. Residual + LayerNorm
        # ---------------------------------------------

        x = self.norm1(
            x + attention_output
        )

        # ---------------------------------------------
        # 6. Feed Forward Network
        # ---------------------------------------------

        ffn_output = self.ffn(x)

        # ---------------------------------------------
        # 7. Residual + LayerNorm
        # ---------------------------------------------

        x = self.norm2(
            x + ffn_output
        )

        # ---------------------------------------------
        # 8. Language Model Head
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
# Input
# =========================================================

input_ids = torch.tensor([
    [2, 5, 7]
])


# =========================================================
# Forward Pass
# =========================================================

logits = model(
    input_ids
)


# =========================================================
# Results
# =========================================================

print("Input IDs:")
print(input_ids)

print("\nInput shape:")
print(input_ids.shape)

print("\nLogits:")
print(logits)

print("\nLogits shape:")
print(logits.shape)