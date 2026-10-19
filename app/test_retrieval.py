from pathlib import Path

import ollama
import chromadb

# Locate the database
folder = Path(__file__).parent
database_path = folder / "chroma_db"

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path=str(database_path)
)

collection = client.get_collection(
    name="study_knowledge"
)


def search_knowledge(question):
    # Convert question into an embedding
    response = ollama.embed(
        model="qwen3-embedding:0.6b",
        input=question
    )

    question_embedding = response["embeddings"][0]

    # Retrieve the closest documents
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=3,
        include=["documents", "distances", "metadatas"]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    print("\nQuestion:", question)
    print("-" * 50)

    for index, (document, distance, metadata) in enumerate(
        zip(documents, distances, metadatas),
        start=1
    ):
        print(f"\nResult {index}")
        print(f"Distance: {distance:.4f}")
        print(f"Document: {document}")
        print(f"Metadata: {metadata}")


# Test questions
test_questions = [
    "What is supervised learning?",
    "What are embeddings?",
    "What does ChromaDB do?",
    "How does RAG work?",
    "What is the capital of France?"
]

for question in test_questions:
    search_knowledge(question)