from pathlib import Path

import ollama
import chromadb

# Locate project files
folder = Path(__file__).parent
filename = folder / "knowledge.txt"
database_path = folder / "chroma_db"

# Read the knowledge document
with open(filename, "r", encoding="utf-8") as file:
    text = file.read().strip()

# Split text into word-based chunks with overlap
def create_chunks(text, chunk_size=40, overlap=10):
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
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start = end - overlap

    return chunks


chunks = create_chunks(text)

# Connect to ChromaDB
client = chromadb.PersistentClient(
    path=str(database_path)
)

collection = client.get_or_create_collection(
    name="study_knowledge"
)

# Remove old chunks so outdated documents don't remain
existing = collection.get()

if existing["ids"]:
    collection.delete(ids=existing["ids"])

# Embed and store the new chunks
for index, chunk in enumerate(chunks):
    response = ollama.embed(
        model="qwen3-embedding:0.6b",
        input=chunk
    )

    embedding = response["embeddings"][0]

    collection.add(
        ids=[str(index)],
        documents=[chunk],
        embeddings=[embedding],
        metadatas=[
            {
                "source": filename.name,
                "chunk_id": index
            }
        ]
    )

    print(f"Stored chunk {index}: {chunk}")

print("\nKnowledge ingestion complete!")
print("Total chunks stored:", len(chunks))
print("Source file:", filename.name)