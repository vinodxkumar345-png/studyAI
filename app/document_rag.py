from pathlib import Path

folder = Path(__file__).parent
filename = folder / "knowledge.txt"


def load_document():
    with open(filename, "r", encoding="utf-8") as file:
        return file.read()


def retrieve_chunks(text, question):
    chunks = text.split(".")
    question_words = question.lower().split()

    results = []

    for chunk in chunks:
        score = sum(
            word in chunk.lower()
            for word in question_words
        )

        if score > 0:
            results.append((score, chunk.strip()))

    results.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return results[:2]


knowledge = load_document()

question = input("Ask a question about your notes: ")

results = retrieve_chunks(
    knowledge,
    question
)

context = "\n".join(
    chunk for score, chunk in results
)

print("\nRetrieved Context")
print("-----------------")

if context:
    print(context)
else:
    print("No relevant information found.")

print("\nRAG Pipeline")
print("------------")
print("Question:", question)
print("Context retrieved successfully.")