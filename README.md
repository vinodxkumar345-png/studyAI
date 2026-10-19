# StudyAI — Local AI Study Assistant

A locally running AI study assistant that helps students understand study materials using Retrieval-Augmented Generation (RAG), semantic search, and large language models.

StudyAI combines FastAPI, Ollama, Qwen3, and ChromaDB to retrieve relevant information from documents and generate student-friendly answers.

## Features

* **Local AI inference:** Generate answers using Ollama and Qwen3.
* **Document-based question answering:** Retrieve relevant information from a local knowledge base.
* **Semantic search:** Find relevant passages using vector embeddings.
* **Persistent vector storage:** Store document embeddings in ChromaDB.
* **Document ingestion:** Process supported study documents into searchable chunks.
* **Source-aware answers:** Return source information associated with retrieved passages.
* **REST API:** Access functionality through FastAPI endpoints.
* **Interactive web interface:** Ask questions through a browser-based interface.
* **Automated testing:** Validate API behavior and retrieval functionality.
* **Privacy-focused architecture:** Process documents and run models locally, subject to the configuration of your environment.

## Technology Stack

| Component       | Technology            |
| --------------- | --------------------- |
| Language        | Python 3.10+          |
| Backend         | FastAPI               |
| API server      | Uvicorn               |
| Language model  | Ollama with Qwen3 4B  |
| Embeddings      | Qwen3 Embedding 0.6B  |
| Vector database | ChromaDB              |
| Frontend        | HTML, CSS, JavaScript |
| Testing         | pytest                |

## Architecture

```text
             Student
                |
                v
       StudyAI Web Interface
                |
                v
          FastAPI Backend
                |
                v
         Student Question
                |
                v
       Query Embedding Model
                |
                v
       ChromaDB Vector Search
                |
                v
        Relevant Text Chunks
                |
                v
          Ollama + Qwen3
                |
                v
        Generated Answer
                |
                v
      Answer and Source Data
```

### How It Works

1. Study documents are processed into text chunks.
2. An embedding model converts the chunks into vector representations.
3. ChromaDB stores the vectors, text, and associated metadata.
4. A student's question is converted into an embedding.
5. Semantic search retrieves relevant document chunks.
6. The language model generates an answer using the available context.
7. FastAPI returns the answer and source information to the frontend.

## Project Structure

```text
StudyAI-publish/
├── README.md
└── app/
    ├── app.py
    ├── ingest.py
    ├── document_rag.py
    ├── evaluate.py
    ├── evaluate_retrieval.py
    ├── evaluation_dataset.json
    ├── knowledge.txt
    ├── requirements.txt
    ├── test_app.py
    ├── test_retrieval.py
    └── static/
        └── index.html
```

The project may contain additional utility scripts and frontend assets.

## Prerequisites

* Python 3.10 or newer
* Ollama installed and running
* Git
* The required Ollama models
* A compatible local environment with sufficient memory and storage

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/vinodxkumar345-png/studyAI.git
cd studyAI
```

If your GitHub repository uses a different URL, use its actual clone URL.

### 2. Create a virtual environment

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r .\app\requirements.txt
```

### 4. Download the required models

Make sure Ollama is installed. Then run:

```powershell
ollama pull qwen3:4b
ollama pull qwen3-embedding:0.6b
```

Start Ollama if it is not already running.

### 5. Prepare the knowledge base

Follow the ingestion instructions provided by `app/ingest.py` and use the document formats supported by the application.

The ChromaDB collection must be initialized and populated according to the application's configuration.

### 6. Start StudyAI

From the repository root:

```powershell
python -m uvicorn app.app:app --reload
```

Open the application in your browser:

http://127.0.0.1:8000/

## API Endpoints

| Endpoint  | Method | Purpose                                   |
| --------- | ------ | ----------------------------------------- |
| `/`       | GET    | Open the web interface                    |
| `/health` | GET    | Check application health                  |
| `/upload` | POST   | Upload supported study documents          |
| `/ask`    | POST   | Submit a question to StudyAI              |
| `/docs`   | GET    | Explore the interactive API documentation |

Consult the running API documentation for the exact request schemas and response formats.

## Running Tests

From the repository root:

```powershell
python -m pytest .\app\test_app.py -v
python -m pytest .\app\test_retrieval.py -v
```

Run the tests before publishing changes. Record the actual results rather than assuming every test passes.

## Configuration and Data

StudyAI uses local models and persistent vector storage. Model names, database paths, retrieval parameters, and upload limits are configured by the application.

Do not commit API keys, passwords, private documents, virtual environments, Python cache files, or generated databases unless there is a specific reason to publish them.

## Current Status

StudyAI is an actively developed local AI study assistant. Features and behavior should be verified against the current code and automated test results.

## Future Improvements

* Better retrieval evaluation and ranking
* Improved answer grounding and source attribution
* More comprehensive automated tests
* Enhanced document management
* Conversation history
* Deployment and installation improvements
* Expanded support for study-document formats

## License

Add a license before distributing the project for reuse. Until a license is selected, all rights remain with the copyright holder.

---

**Built with Python, FastAPI, Ollama, Qwen3, and ChromaDB.**
