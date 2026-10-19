import ollama

response = ollama.embed(
    model="qwen3-embedding:0.6b",
    input="Machine learning learns patterns from data."
)

embedding = response["embeddings"][0]

print("Embedding created!")
print("Number of values:", len(embedding))
print("First 10 values:", embedding[:10])