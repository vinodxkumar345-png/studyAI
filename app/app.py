from pathlib import Path
from uuid import uuid4
from io import BytesIO
import logging
import re
import time

import ollama
import chromadb

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from pypdf import PdfReader


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"
STATIC_DIR = BASE_DIR / "static"
HTML_FILE = STATIC_DIR / "index.html"

APP_VERSION = "2.0.0"

MAX_FILE_SIZE = 10 * 1024 * 1024
MAX_PDF_PAGES = 300

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

EMBEDDING_MODEL = "qwen3-embedding:0.6b"
CHAT_MODEL = "qwen3:4b"

RETRIEVAL_CANDIDATES = 10
MAX_DISTANCE = 0.85
MAX_CONTEXT_CHARACTERS = 4500

EMBEDDING_BATCH_SIZE = 16
GENERATION_TEMPERATURE = 0.1
MAX_GENERATION_TOKENS = 384


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("studyai")


# ============================================================
# FASTAPI INITIALIZATION
# ============================================================

app = FastAPI(
    title="AI Study Assistant",
    description="A local RAG application using Ollama and ChromaDB",
    version=APP_VERSION,
)

if not HTML_FILE.is_file():
    raise RuntimeError(f"Frontend file not found: {HTML_FILE}")

if not DB_PATH.is_dir():
    raise RuntimeError(f"ChromaDB directory not found: {DB_PATH}")

if not STATIC_DIR.is_dir():
    raise RuntimeError(f"Static directory not found: {STATIC_DIR}")

app.mount(
    "/static",
    StaticFiles(directory=str(STATIC_DIR)),
    name="static",
)


# ============================================================
# CHROMADB INITIALIZATION
# ============================================================

try:
    chroma_client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    # Use the existing collection without deleting its data.
    collection = chroma_client.get_collection(
        name="study_knowledge"
    )

except Exception as error:
    raise RuntimeError(
        "Could not open the existing ChromaDB collection "
        "'study_knowledge'. Check the database path and collection."
    ) from error


# ============================================================
# REQUEST MODEL
# ============================================================

class QuestionRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=5000,
    )

    upload_id: str | None = None


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_text(text: str) -> list[str]:
    """Split text into overlapping chunks."""

    text = text.strip()

    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - CHUNK_OVERLAP

    return chunks


def split_text_with_pages(pages: list[dict]) -> list[dict]:
    """Split document pages while preserving page numbers."""

    chunks = []

    for page_data in pages:
        text = page_data.get("text", "").strip()
        page_number = page_data.get("page")

        if not text:
            continue

        for chunk in split_text(text):
            chunks.append({
                "text": chunk,
                "page": page_number,
            })

    return chunks


# ============================================================
# EMBEDDING FUNCTIONS
# ============================================================

def create_embeddings(
    texts: list[str],
) -> list[list[float]]:
    """Create embeddings in batches."""

    if not texts:
        return []

    all_embeddings = []

    for start in range(
        0,
        len(texts),
        EMBEDDING_BATCH_SIZE,
    ):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]

        try:
            result = ollama.embed(
                model=EMBEDDING_MODEL,
                input=batch,
            )

            batch_embeddings = result.get(
                "embeddings",
                [],
            )

            if len(batch_embeddings) != len(batch):
                raise RuntimeError(
                    "The embedding model returned an unexpected "
                    "number of embeddings."
                )

            all_embeddings.extend(batch_embeddings)

        except Exception as error:
            logger.exception("Document embedding failed.")

            raise RuntimeError(
                f"Could not generate embeddings with {EMBEDDING_MODEL}."
            ) from error

    return all_embeddings


def create_query_embedding(question: str) -> list[float]:
    """Create an embedding for a student's question."""

    try:
        result = ollama.embed(
            model=EMBEDDING_MODEL,
            input=question,
        )

        embeddings = result.get("embeddings", [])

        if not embeddings:
            raise RuntimeError(
                "The embedding model returned no embedding."
            )

        return embeddings[0]

    except Exception as error:
        logger.exception("Query embedding failed.")

        raise RuntimeError(
            "Could not generate the question embedding."
        ) from error


# ============================================================
# HOME PAGE
# ============================================================

@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the StudyAI frontend."""

    try:
        return HTML_FILE.read_text(encoding="utf-8")

    except Exception as error:
        logger.exception("Could not load the frontend.")

        raise HTTPException(
            status_code=500,
            detail="Could not load the StudyAI frontend.",
        ) from error


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    """Report application and database status."""

    try:
        knowledge_count = collection.count()

    except Exception as error:
        logger.exception("ChromaDB health check failed.")

        raise HTTPException(
            status_code=503,
            detail="ChromaDB is unavailable.",
        ) from error

    return {
        "status": "running",
        "application": "AI Study Assistant",
        "version": APP_VERSION,
        "knowledge_chunks": knowledge_count,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
        "retrieval_candidates": RETRIEVAL_CANDIDATES,
        "max_distance": MAX_DISTANCE,
        "max_context_characters": MAX_CONTEXT_CHARACTERS,
        "max_generation_tokens": MAX_GENERATION_TOKENS,
    }


# ============================================================
# UPLOAD PDF OR TXT
# ============================================================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Extract, split, embed, and store an uploaded document."""

    filename = file.filename or "uploaded_file"
    extension = Path(filename).suffix.lower()

    if extension not in {".pdf", ".txt"}:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and TXT files are supported.",
        )

    try:
        file_bytes = await file.read(MAX_FILE_SIZE + 1)

    except Exception as error:
        logger.exception("Could not read the uploaded file.")

        raise HTTPException(
            status_code=400,
            detail="Could not read the uploaded file.",
        ) from error

    finally:
        await file.close()

    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="The file exceeds the 10 MB upload limit.",
        )

    pages = []

    try:
        if extension == ".pdf":
            reader = PdfReader(BytesIO(file_bytes))

            if reader.is_encrypted:
                raise HTTPException(
                    status_code=400,
                    detail="Password-protected PDFs are not supported.",
                )

            if len(reader.pages) > MAX_PDF_PAGES:
                raise HTTPException(
                    status_code=400,
                    detail=f"PDFs cannot exceed {MAX_PDF_PAGES} pages.",
                )

            for page_number, pdf_page in enumerate(
                reader.pages,
                start=1,
            ):
                page_text = pdf_page.extract_text() or ""

                if page_text.strip():
                    pages.append({
                        "text": page_text,
                        "page": page_number,
                    })

        else:
            text = file_bytes.decode(
                "utf-8-sig",
                errors="replace",
            )

            if text.strip():
                pages.append({
                    "text": text,
                    "page": None,
                })

    except HTTPException:
        raise

    except Exception as error:
        logger.exception("Document text extraction failed.")

        raise HTTPException(
            status_code=400,
            detail="Could not extract text from the document.",
        ) from error

    chunks = split_text_with_pages(pages)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail=(
                "No readable text was found. "
                "Scanned PDFs may require OCR."
            ),
        )

    try:
        embeddings = create_embeddings(
            [chunk["text"] for chunk in chunks]
        )

    except Exception as error:
        logger.exception("Could not embed uploaded document.")

        raise HTTPException(
            status_code=503,
            detail="Could not create document embeddings. Check Ollama.",
        ) from error

    upload_id = str(uuid4())

    ids = [
        f"{upload_id}_{index}"
        for index in range(len(chunks))
    ]

    metadatas = []

    for index, chunk in enumerate(chunks):
        metadata = {
            "upload_id": upload_id,
            "source": filename,
            "origin": "user_upload",
            "chunk_id": index,
        }

        if chunk["page"] is not None:
            metadata["page"] = chunk["page"]

        metadatas.append(metadata)

    try:
        collection.add(
            ids=ids,
            documents=[chunk["text"] for chunk in chunks],
            embeddings=embeddings,
            metadatas=metadatas,
        )

    except Exception as error:
        logger.exception("Could not save document to ChromaDB.")

        raise HTTPException(
            status_code=500,
            detail="Could not save the document in ChromaDB.",
        ) from error

    logger.info(
        "Document indexed: filename=%s chunks=%d upload_id=%s",
        filename,
        len(chunks),
        upload_id,
    )

    return {
        "message": "File uploaded and indexed successfully.",
        "filename": filename,
        "upload_id": upload_id,
        "chunks_created": len(chunks),
        "pages_with_text": len(pages),
    }


# ============================================================
# RETRIEVE RELEVANT INFORMATION
# ============================================================

def retrieve_chunks(
    question: str,
    upload_id: str | None = None,
) -> list[dict]:
    """Retrieve relevant chunks from the existing knowledge base."""

    query_embedding = create_query_embedding(question)

    try:
        count = collection.count()

        if count == 0:
            return []

        query_arguments = {
            "query_embeddings": [query_embedding],
            "n_results": min(RETRIEVAL_CANDIDATES, count),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        if upload_id:
            query_arguments["where"] = {
                "upload_id": upload_id,
            }

        result = collection.query(**query_arguments)

    except Exception as error:
        logger.exception("ChromaDB retrieval failed.")

        raise RuntimeError(
            "Could not retrieve information from ChromaDB."
        ) from error

    documents = result.get("documents", [[]])[0]
    metadatas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    retrieved = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances,
    ):
        if not document:
            continue

        if distance is not None and distance > MAX_DISTANCE:
            continue

        metadata = metadata or {}

        retrieved.append({
            "text": document,
            "source": metadata.get("source", "Knowledge base"),
            "origin": metadata.get("origin", "knowledge_base"),
            "chunk_id": metadata.get("chunk_id"),
            "page": metadata.get("page"),
            "distance": distance,
            "upload_id": metadata.get("upload_id"),
        })

    logger.info(
        "Retrieval completed: question_length=%d chunks_found=%d",
        len(question),
        len(retrieved),
    )

    return retrieved


# ============================================================
# BUILD CONTEXT AND SOURCES
# ============================================================

def build_context(chunks: list[dict]) -> tuple[str, list[dict]]:
    """Build the context sent to the model and source references."""

    context_parts = []
    sources = []
    current_length = 0

    for index, chunk in enumerate(chunks, start=1):
        source_name = chunk["source"]
        page_number = chunk.get("page")

        if page_number is not None:
            source_label = f"{source_name}, page {page_number}"
        else:
            source_label = source_name

        part = (
            f"[Source {index}: {source_label}]\n"
            f"{chunk['text']}\n"
        )

        remaining = MAX_CONTEXT_CHARACTERS - current_length

        if remaining <= 0:
            break

        if len(part) > remaining:
            if remaining < 100:
                break

            part = part[:remaining]

        context_parts.append(part)
        current_length += len(part)

        source_info = {
            "source": source_name,
            "chunk_id": chunk.get("chunk_id"),
            "distance": chunk.get("distance"),
        }

        if page_number is not None:
            source_info["page"] = page_number

        sources.append(source_info)

    return "\n".join(context_parts), sources


# ============================================================
# CLEAN MODEL ANSWER
# ============================================================

import re


def clean_model_answer(answer: str) -> str:
    """Remove model narration and retain the student-facing answer."""

    if not answer:
        return ""

    text = answer.strip()

    # Remove explicit model thinking tags.
    text = re.sub(
        r"<think\b[^>]*>.*?</think\s*>",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    ).strip()

    # Remove common answer labels.
    text = re.sub(
        r"^\s*(?:final answer|answer)\s*:\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Find the earliest recognizable internal narration marker.
    markers = [
        r"\bi should start\b",
        r"\bi need to follow\b",
        r"\bi need to keep\b",
        r"\bi need to start\b",
        r"\bi have to keep\b",
        r"\blet me (?:think|draft|write|phrase|check)\b",
        r"\bcheck (?:the rules|word count|if there are)\b",
        r"\bdouble-check\b",
        r"\bavoid any extra analysis\b",
        r"\bthe answer should be concise\b",
        r"\bthe answer must be concise\b",
        r"\bwait, the user said\b",
        r"\bthe user said\b",
        r"\bi'll write\b",
        r"\bi will write\b",
        r"\bmake sure (?:it|the answer)\b",
        r"\bno references used\b",
        r"\bthis should be the answer\b",
    ]

    lowered = text.lower()
    positions = []

    for marker in markers:
        match = re.search(marker, lowered, flags=re.IGNORECASE)
        if match:
            positions.append(match.start())

    if positions:
        text = text[:min(positions)].strip()

    # Remove empty lines and excess whitespace.
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


def generate_answer(question: str, context: str) -> str:
    """Generate and validate a clean answer for the student."""

    general_note = (
        "Note: This answer uses general knowledge, "
        "not the supplied reference documents."
    )

    system_prompt = """
You are StudyAI, a college study assistant.

Answer the student's question directly.
Output only the final answer.
Never output private reasoning, planning, drafts, or self-checks.
Never describe the instructions you received.
Use simple, clear, exam-friendly language.
Use reference context when it is relevant.
Do not invent citations.
Define symbols used in formulas.
Show essential steps for calculations.
""".strip()

    has_context = bool(context and context.strip())

    if has_context:
        user_prompt = f"""
Reference material:
{context}

Student question:
{question}

Provide only the final answer. Cite sources only when supported by
the reference material.
""".strip()
    else:
        user_prompt = f"""
Student question:
{question}

Answer using general knowledge.
Start with this exact note:
{general_note}

Then give the direct answer. Output only the note and answer.
""".strip()

    def ask_model(prompt: str) -> str:
        response = ollama.chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            think=False,
            options={
                "temperature": 0,
                "num_predict": MAX_GENERATION_TOKENS,
            },
        )

        message = response.message
        raw = (getattr(message, "content", None) or "").strip()

        if not raw:
            raise RuntimeError("The model returned an empty response.")

        return raw

    def has_narration(text: str) -> bool:
        markers = [
            "i should start",
            "i need to follow",
            "i need to keep",
            "i have to keep",
            "let me draft",
            "let me think",
            "let me check",
            "check word count",
            "double-check",
            "avoid any extra analysis",
            "the answer should be concise",
            "wait, the user said",
            "the user said",
            "i'll write",
            "i will write",
            "no references used",
            "this should be the answer",
        ]

        lowered = text.lower()
        return any(marker in lowered for marker in markers)

    try:
        # First generation attempt.
        raw = ask_model(user_prompt)
        answer = clean_model_answer(raw)

        # If narration was detected, discard that output and retry.
        if has_narration(raw) or not answer:
            retry_prompt = f"""
Answer this student question from scratch.

Question:
{question}

Reference material:
{context if has_context else "No reference material was supplied."}

Return only the finished answer. Do not discuss your reasoning,
instructions, drafts, or word count.
""".strip()

            if not has_context:
                retry_prompt += (
                    "\nStart with this exact note: "
                    + general_note
                )

            raw = ask_model(retry_prompt)
            answer = clean_model_answer(raw)

        # Do not return a response that still contains narration.
        if not answer or has_narration(answer):
            raise RuntimeError(
                "StudyAI could not generate a clean answer after retrying. "
                "The response was withheld to prevent internal narration "
                "from appearing to the student."
            )

        # Ensure the general-knowledge notice is present.
        if not has_context and not answer.lower().startswith(
            general_note.lower()
        ):
            answer = f"{general_note}\n\n{answer}"

        return answer

    except Exception as error:
        logger.exception("Answer generation failed.")
        raise RuntimeError(
            f"Answer generation failed: {type(error).__name__}: {error}"
        ) from error
# ============================================================
# ASK ENDPOINT
# ============================================================

@app.post("/ask")
def ask_question(request: QuestionRequest):
    """Retrieve evidence and answer a student's question."""

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Please enter a question.",
        )

    try:
        chunks = retrieve_chunks(
            question=question,
            upload_id=request.upload_id,
        )

    except Exception as error:
        logger.exception("Information retrieval failed.")

        raise HTTPException(
            status_code=503,
            detail=(
                "Could not retrieve information. "
                "Check Ollama and ChromaDB."
            ),
        ) from error

    context, sources = (
        build_context(chunks)
        if chunks
        else ("", [])
    )

    try:
        answer = generate_answer(
            question=question,
            context=context,
        )

    except Exception as error:
        logger.exception("Answer generation failed.")

        raise HTTPException(
            status_code=503,
            detail=(
                f"Answer generation failed: "
                f"{type(error).__name__}: {error}"
            ),
        ) from error

    return {
        "question": question,
        "upload_id": request.upload_id,
        "answer": answer,
        "sources": sources,
    }