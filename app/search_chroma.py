from pathlib import Path
import ollama
import chromadb


# Connect to the existing database
folder = Path(__file__).parent
database_path = folder / "chroma_db"

client = chromadb.PersistentClient(
    path=str(database_path)
)


# Open the existing collection
collection = client.get_collection(
    name="study_knowledge"
)


# Ask the user
question = input("Ask a question: ")


# Create an embedding for ONLY the question
response = ollama.embed(
    model="qwen3-embedding:0.6b",
    input=question
)

question_embedding = response["embeddings"][0]


# Search ChromaDB
results = collection.query(
    query_embeddings=[question_embedding],
    n_results=2
)


# Display results
print("\nRelevant Knowledge")
print("------------------")

documents = results["documents"][0]
distances = results["distances"][0]

for document, distance in zip(documents, distances):

    print("\nDistance:", round(distance, 4))
    print("Text:", document)