import numpy as np

vector_a = np.array([1.0, 2.0, 3.0])
vector_b = np.array([1.1, 2.1, 3.1])
vector_c = np.array([-2.0, 0.0, 1.0])


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


print("Vector A:")
print(vector_a)

print("\nVector B:")
print(vector_b)

print("\nVector C:")
print(vector_c)

print("\nCosine similarity:")

print("A vs B:")
print(cosine_similarity(vector_a, vector_b))

print("\nA vs C:")
print(cosine_similarity(vector_a, vector_c))