from pathlib import Path
import ollama
import math


folder = Path(__file__).parent
filename = folder / "knowledge.txt"


def load_document():
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def create_embedding(text):
    response = ollama.embed(
        model="qwen3-embedding:0.6b",
        input=text
    )

    return response["embeddings"][0]


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


def create_chunks(text):
    chunks = text.split(".")

    return [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]


knowledge = load_document()

chunks = create_chunks(knowledge)

question = input("Ask a question: ")

question_embedding = create_embedding(question)

results = []

for chunk in chunks:

    chunk_embedding = create_embedding(chunk)

    similarity = cosine_similarity(
        question_embedding,
        chunk_embedding
    )

    results.append((similarity, chunk))


results.sort(
    key=lambda item: item[0],
    reverse=True
)


print("\nSemantic Search Results")
print("----------------------")

for score, chunk in results:

    print("\nSimilarity:", round(score, 4))
    print("Text:", chunk)