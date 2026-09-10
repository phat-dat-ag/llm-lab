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
        # 3. Token + Position
        # ---------------------------------------------

        x = token_vectors + position_vectors

        # ---------------------------------------------
        # 4. Self-Attention
        # ---------------------------------------------

        attention_output, attention_weights = (
            self.attention(
                x,
                x,
                x,
                need_weights=True
            )
        )

        # ---------------------------------------------
        # 5. Residual + LayerNorm
        # ---------------------------------------------

        x = self.norm1(
            x + attention_output
        )

        # ---------------------------------------------
        # 6. Feed Forward
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

        logits = self.lm_head(x)

        return (
            logits,
            attention_weights
        )


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
# Parameter Count
# =========================================================

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Total parameters:")
print(total_parameters)


# =========================================================
# Parameter Details
# =========================================================

print("\nParameter details:")

for name, parameter in model.named_parameters():

    print(
        f"{name}: {tuple(parameter.shape)}"
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

with torch.no_grad():

    logits, attention_weights = model(
        input_ids
    )


# =========================================================
# Inspect Logits
# =========================================================

print("\nInput IDs:")
print(input_ids)

print("\nInput shape:")
print(input_ids.shape)

print("\nLogits shape:")
print(logits.shape)

print("\nLogits from last position:")
print(logits[:, -1, :])


# =========================================================
# Greedy Prediction
# =========================================================

next_token = torch.argmax(
    logits[:, -1, :],
    dim=-1
)

print("\nPredicted next token:")
print(next_token)


# =========================================================
# Attention Weights
# =========================================================

print("\nAttention weights:")
print(attention_weights)

print("\nAttention weights shape:")
print(attention_weights.shape)


# =========================================================
# Generation Experiment
# =========================================================

generated_ids = input_ids.clone()

num_new_tokens = 5

print("\nGeneration:")

with torch.no_grad():

    for step in range(num_new_tokens):

        logits, _ = model(
            generated_ids
        )

        next_token_logits = (
            logits[:, -1, :]
        )

        next_token = torch.argmax(
            next_token_logits,
            dim=-1,
            keepdim=True
        )

        generated_ids = torch.cat(
            [
                generated_ids,
                next_token
            ],
            dim=1
        )

        print(
            f"Step {step + 1}: "
            f"{generated_ids}"
        )


# =========================================================
# Final Result
# =========================================================

print("\nFinal generated IDs:")
print(generated_ids)

print("\nFinal generated shape:")
print(generated_ids.shape)