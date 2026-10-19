# Local AI Study Assistant

A Retrieval-Augmented Generation (RAG) application that answers
questions using a local knowledge base.

## Features

- Local language model using Ollama and Qwen3
- Text embeddings using Qwen3 Embedding
- Vector search using ChromaDB
- FastAPI backend
- Source metadata for retrieved documents
- Automated API tests using pytest

## Architecture

1. Knowledge documents are loaded and divided into chunks.
2. Each chunk is converted into an embedding.
3. ChromaDB stores the embeddings and document metadata.
4. A user's question is converted into an embedding.
5. ChromaDB retrieves relevant documents.
6. The local language model generates an answer using the retrieved context.
7. FastAPI returns the answer and source information.

## Technologies

- Python 3.10
- Ollama
- Qwen3 4B
- Qwen3 Embedding 0.6B
- ChromaDB
- FastAPI
- Uvicorn
- pytest
- HTTPX

## Requirements

- Python 3.10
- Ollama installed
- The required Ollama models downloaded

## Install Python dependencies

```bash
py -3.10 -m pip install ollama chromadb fastapi uvicorn pytest httpx