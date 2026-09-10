import torch
import torch.nn as nn


# =========================================================
# Configuration
# =========================================================

vocabulary_size = 10
embedding_dimension = 8
hidden_dimension = 16
max_sequence_length = 5


# =========================================================
# Tiny Transformer Block
# =========================================================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embedding_dimension,
        hidden_dimension,
        num_heads
    ):
        super().__init__()

        # Multi-Head Self-Attention
        self.attention = nn.MultiheadAttention(
            embed_dim=embedding_dimension,
            num_heads=num_heads,
            batch_first=True
        )

        # Feed Forward Network
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

        # Layer Normalization
        self.norm1 = nn.LayerNorm(
            embedding_dimension
        )

        self.norm2 = nn.LayerNorm(
            embedding_dimension
        )


    def forward(self, x):

        # ---------------------------------------------
        # 1. Self-Attention
        # ---------------------------------------------

        attention_output, attention_weights = (
            self.attention(
                x,
                x,
                x
            )
        )


        # ---------------------------------------------
        # 2. Residual + LayerNorm
        # ---------------------------------------------

        x = self.norm1(
            x + attention_output
        )


        # ---------------------------------------------
        # 3. Feed Forward Network
        # ---------------------------------------------

        ffn_output = self.ffn(x)


        # ---------------------------------------------
        # 4. Residual + LayerNorm
        # ---------------------------------------------

        x = self.norm2(
            x + ffn_output
        )


        return x


# =========================================================
# Token + Position Embedding
# =========================================================

token_embedding = nn.Embedding(
    vocabulary_size,
    embedding_dimension
)

position_embedding = nn.Embedding(
    max_sequence_length,
    embedding_dimension
)


# =========================================================
# Input
# =========================================================

input_ids = torch.tensor([
    [2, 5, 7]
])


sequence_length = input_ids.shape[1]

positions = torch.arange(
    sequence_length
).unsqueeze(0)


# =========================================================
# Create Input Representation
# =========================================================

token_vectors = token_embedding(
    input_ids
)

position_vectors = position_embedding(
    positions
)

x = token_vectors + position_vectors


# =========================================================
# Create Transformer Block
# =========================================================

transformer_block = TransformerBlock(
    embedding_dimension=embedding_dimension,
    hidden_dimension=hidden_dimension,
    num_heads=2
)


# =========================================================
# Forward Pass
# =========================================================

output = transformer_block(x)


# =========================================================
# Results
# =========================================================

print("Input IDs:")
print(input_ids)

print("\nInput IDs shape:")
print(input_ids.shape)

print("\nInput representation shape:")
print(x.shape)

print("\nTransformer output:")
print(output)

print("\nTransformer output shape:")
print(output.shape)