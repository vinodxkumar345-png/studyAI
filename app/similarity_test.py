import math


def cosine_similarity(vector_a, vector_b):
    dot_product = 0
    magnitude_a = 0
    magnitude_b = 0

    for a, b in zip(vector_a, vector_b):
        dot_product += a * b
        magnitude_a += a * a
        magnitude_b += b * b

    magnitude_a = math.sqrt(magnitude_a)
    magnitude_b = math.sqrt(magnitude_b)

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot_product / (magnitude_a * magnitude_b)


vector_1 = [1, 2, 3]
vector_2 = [1, 2, 3]

score = cosine_similarity(vector_1, vector_2)

print("Similarity:", score)