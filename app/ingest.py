
from pathlib import Path

import chromadb
import ollama

BASE_DIR = Path(__file__).resolve().parent
KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"
DATABASE_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "study_knowledge"
EMBEDDING_MODEL = "qwen3-embedding:0.6b"

CHUNK_SIZE = 40
CHUNK_OVERLAP = 10


def create_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size"
        )

    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start = end - overlap

    return chunks


def main():
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_FILE}"
        )

    text = KNOWLEDGE_FILE.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("knowledge.txt is empty.")

    chunks = create_chunks(text)

    client = chromadb.PersistentClient(
        path=str(DATABASE_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Delete only previous built-in knowledge-base records.
    # Uploaded documents are preserved.
    existing = collection.get(include=["metadatas"])

    ids_to_replace = []

    for record_id, metadata in zip(
        existing["ids"],
        existing["metadatas"] or []
    ):
        if not metadata:
            continue

        is_old_builtin_record = (
            metadata.get("source") == KNOWLEDGE_FILE.name
            and not metadata.get("upload_id")
        )

        is_new_builtin_record = (
            metadata.get("origin") == "knowledge_base"
        )

        if is_old_builtin_record or is_new_builtin_record:
            ids_to_replace.append(record_id)

    if ids_to_replace:
        collection.delete(ids=ids_to_replace)

    # Create embeddings and insert the refreshed built-in knowledge.
    for index, chunk in enumerate(chunks):
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=chunk
        )

        embedding = response["embeddings"][0]

        collection.add(
            ids=[f"knowledge_base_{index}"],
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[
                {
                    "source": KNOWLEDGE_FILE.name,
                    "chunk_id": index,
                    "origin": "knowledge_base",
                }
            ],
        )

    print("Knowledge-base ingestion completed.")
    print(f"Chunks processed: {len(chunks)}")
    print(f"Total collection records: {collection.count()}")


if __name__ == "__main__":
    main()

