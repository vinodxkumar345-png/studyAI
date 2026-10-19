from pathlib import Path

import chromadb
import ollama


# --------------------------------
# 1. Connect to ChromaDB
# --------------------------------

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chroma_db"

client = chromadb.PersistentClient(
    path=str(DB_PATH)
)

collection = client.get_collection(
    name="study_knowledge"
)


# --------------------------------
# 2. Configure retrieval
# --------------------------------

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
TOP_K = 5
MAX_DISTANCE = 1.5


# --------------------------------
# 3. Search for relevant sources
# --------------------------------

def retrieve_sources(question: str):

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if collection.count() == 0:
        raise RuntimeError(
            "Knowledge base is empty. Run ingest.py first."
        )

    embedding_response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question
    )

    question_embedding = embedding_response["embeddings"][0]

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=min(TOP_K, collection.count()),
        include=[
            "documents",
            "distances",
            "metadatas"
        ]
    )

    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    sources = []

    for document, distance, metadata in zip(
        documents,
        distances,
        metadatas
    ):
        if distance <= MAX_DISTANCE:
            metadata = metadata or {}

            sources.append({
                "text": document,
                "distance": round(distance, 4),
                "source": metadata.get("source", "Unknown source"),
                "chunk_id": metadata.get("chunk_id", "Unknown")
            })

    return sources


# --------------------------------
# 4. Display and verify sources
# --------------------------------

def main():

    question = input("Enter your question: ").strip()

    try:
        sources = retrieve_sources(question)

    except (ValueError, RuntimeError) as error:
        print(f"\nError: {error}")
        return

    print("\nQuestion:", question)
    print("Relevant sources found:", len(sources))

    if not sources:
        print("\nNo sufficiently relevant sources found.")
        return

    for index, source in enumerate(sources, start=1):
        print(f"\n--- Source {index} ---")
        print("Document:", source["source"])
        print("Chunk ID:", source["chunk_id"])
        print("Distance:", source["distance"])
        print("Content:", source["text"])


if __name__ == "__main__":
    main()