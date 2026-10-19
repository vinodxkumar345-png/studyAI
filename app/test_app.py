from fastapi.testclient import TestClient
from unittest.mock import patch
import ollama

from app import app

client = TestClient(app)


# --------------------------------------------------
# 1. Home page
# --------------------------------------------------

def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers.get("content-type", "")


# --------------------------------------------------
# 2. Question validation
# --------------------------------------------------

def test_ask_endpoint_rejects_empty_question():
    response = client.post(
        "/ask",
        json={"question": "   "}
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Please enter a question."


def test_ask_endpoint_rejects_missing_question():
    response = client.post(
        "/ask",
        json={}
    )

    assert response.status_code == 422


def test_ask_endpoint_rejects_invalid_question_type():
    response = client.post(
        "/ask",
        json={"question": 123}
    )

    assert response.status_code == 422


# --------------------------------------------------
# 3. Knowledge-base validation
# --------------------------------------------------

def test_ask_endpoint_rejects_empty_knowledge_base():
    with patch("app.collection") as mock_collection:
        mock_collection.count.return_value = 0

        response = client.post(
            "/ask",
            json={"question": "What is supervised learning?"}
        )

    assert response.status_code == 503
    assert "knowledge base is empty" in response.json()["detail"].lower()


# --------------------------------------------------
# 4. Ollama request error handling
# --------------------------------------------------

def test_ask_endpoint_handles_ollama_request_error():
    with patch("app.collection") as mock_collection:
        mock_collection.count.return_value = 1

        with patch("app.create_query_embedding") as mock_embedding:
            mock_embedding.side_effect = ollama.RequestError(
                "Ollama is unavailable"
            )

            response = client.post(
                "/ask",
                json={"question": "What is supervised learning?"}
            )

    assert response.status_code == 503
    
    assert response.json()["detail"] == (
    "Could not retrieve information. "
    "Check Ollama, the embedding model, and ChromaDB. "
    "See the terminal logs for details."
)




# --------------------------------------------------
# 5. Unexpected error handling
# --------------------------------------------------

def test_ask_endpoint_handles_unexpected_error():
    with patch("app.collection") as mock_collection:
        mock_collection.count.return_value = 1

        with patch("app.create_query_embedding") as mock_embedding:
            mock_embedding.side_effect = ValueError(
                "Unexpected test error"
            )

            response = client.post(
                "/ask",
                json={"question": "What is supervised learning?"}
            )

    assert response.status_code == 503
    assert response.json()["detail"] == (
        "Could not retrieve information. Check Ollama, the embedding model, and ChromaDB. See the terminal logs for details."
    )


# --------------------------------------------------
# 6. Health endpoint
# --------------------------------------------------

def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "running"
    assert data["version"] == "2.1.0"