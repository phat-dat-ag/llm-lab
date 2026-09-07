import numpy as np

np.random.seed(42)

tokens = [
    "hello",
    "world",
    "cat",
    "dog",
    "japanese",
]

vocabulary_size = len(tokens)
embedding_dimension = 8

embedding_matrix = np.random.randn(
    vocabulary_size,
    embedding_dimension
)

token_to_id = {
    token: token_id
    for token_id, token in enumerate(tokens)
}


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


print("Vocabulary:")
print(token_to_id)

print("\nEmbedding Matrix:")
print(embedding_matrix)

print("\nToken vectors:")

for token, token_id in token_to_id.items():
    vector = embedding_matrix[token_id]

    print(f"\n{token} (ID={token_id})")
    print(vector)

print("\nSimilarity experiments:")

pairs = [
    ("hello", "world"),
    ("cat", "dog"),
    ("hello", "cat"),
]

for token_a, token_b in pairs:
    vector_a = embedding_matrix[token_to_id[token_a]]
    vector_b = embedding_matrix[token_to_id[token_b]]

    similarity = cosine_similarity(vector_a, vector_b)

    print(
        f"{token_a} vs {token_b}: "
        f"{similarity:.4f}"
    )