from pathlib import Path

import chromadb
import ollama

# 1. Locate the ChromaDB database
BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "chroma_db"

# 2. Connect to the existing database
client = chromadb.PersistentClient(path=str(DB_PATH))

# 3. Open the existing knowledge collection
collection = client.get_collection(name="study_knowledge")

# 4. Questions and expected topics
test_cases = [
    {
        "question": "What is supervised learning?",
        "expected_keywords": ["supervised learning"],
    },
    {
        "question": "What is ChromaDB?",
        "expected_keywords": ["chromadb"],
    },
    {
        "question": "What is RAG?",
        "expected_keywords": ["retrieval-augmented generation", "rag system"],
    },
]

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
TOP_K = 3


def evaluate_retrieval():
    passed = 0

    for test in test_cases:
        question = test["question"]

        # Convert the question into an embedding
        result = ollama.embed(
            model=EMBEDDING_MODEL,
            input=question,
        )
        query_embedding = result["embeddings"][0]

        # Retrieve the three most relevant chunks
        search_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(TOP_K, collection.count()),
        )

        documents = search_results["documents"][0]
        combined_text = " ".join(documents).lower()

        # Check whether an expected topic appears in the retrieved text
        found = any(
            keyword.lower() in combined_text
            for keyword in test["expected_keywords"]
        )

        if found:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"\nQuestion: {question}")
        print(f"Result: {status}")
        print(f"Retrieved chunks: {len(documents)}")

        for index, document in enumerate(documents, start=1):
            print(f"  Chunk {index}: {document[:180]}...")

    print("\n" + "=" * 40)
    print(f"Retrieval checks passed: {passed}/{len(test_cases)}")
    print("=" * 40)


if __name__ == "__main__":
    evaluate_retrieval()