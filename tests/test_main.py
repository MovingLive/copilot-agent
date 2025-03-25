"""Tests unitaires pour le module main."""

import os
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.embedding_service import embed_text, generate_query_vector
from app.services.faiss_service import load_index, retrieve_similar_documents

@pytest.fixture(name="client")
def test_client():
    """Fixture pour le client de test FastAPI."""
    return TestClient(app)

@pytest.fixture
def mock_env_vars():
    """Fixture pour configurer les variables d'environnement de test."""
    with patch.dict(
        os.environ,
        {
            "ENV": "test",
            "S3_BUCKET_NAME": "test-bucket",
            "AWS_REGION": "ca-central-1",
            "COPILOT_API_URL": "https://test-api.com",
            "COPILOT_TOKEN": "test-token",
        },
    ):
        yield

def test_root(client):
    """Teste l'endpoint racine de l'API."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API Copilot LLM!"}

def test_embed_text():
    """Teste la fonction d'embedding de texte."""
    text = "Test text"
    embedding = embed_text(text)

    assert len(embedding) == 384
    assert all(0 <= x <= 1 for x in embedding)
    assert embed_text(text) == embedding

def test_generate_query_vector():
    """Teste la génération du vecteur de requête."""
    with patch("app.services.embedding_service.embed_text", return_value=[0.1] * 384):
        query = "test query"
        query_vector = generate_query_vector(query)

        assert query_vector.shape[1] == 384
        assert query_vector.dtype == "float32"

def test_retrieve_similar_documents_empty():
    """Teste la récupération de documents avec un index vide."""
    with patch("app.services.faiss_service._state.get_state", return_value=(None, {})):
        docs = retrieve_similar_documents("test query", k=3)
        assert len(docs) == 0

@pytest.mark.asyncio
async def test_query_success(client):
    """Teste l'endpoint principal de l'API pour les requêtes."""
    mock_docs = [{"content": "Test content", "metadata": {"file": "test.md"}}]

    with (patch("app.services.faiss_service.retrieve_similar_documents", return_value=mock_docs),
          patch("app.api.copilot.get_github_user", new_callable=AsyncMock) as mock_user):

        mock_user.return_value = "test_user"

        response = client.post(
            "/api",
            headers={"x-github-token": "test-token"},
            json={
                "messages": [{"content": "Test question", "role": "user"}],
                "copilot_references": "Additional context"
            }
        )

        assert response.status_code == 200

@pytest.mark.asyncio
async def test_query_invalid(client):
    """Teste l'endpoint principal avec une requête invalide."""
    response = client.post("/api", json={})
    assert response.status_code == 401

def test_load_faiss_index_local():
    """Teste le chargement de l'index FAISS en environnement local."""
    with (patch("app.services.faiss_service._get_local_path") as mock_path,
          patch("app.services.faiss_service.is_local_environment", return_value=True)):

        mock_path.return_value = "/test/path/index.faiss"
        index, doc_store = load_index()

        assert index is None  # Car le fichier n'existe pas en test
        assert isinstance(doc_store, dict)

def test_load_faiss_index_s3():
    """Teste le chargement de l'index FAISS depuis S3."""
    with (patch("app.services.faiss_service._get_local_path") as mock_path,
          patch("app.services.faiss_service.is_local_environment", return_value=False),
          patch("app.services.faiss_service._download_from_s3")):

        mock_path.return_value = "/tmp/index.faiss"
        index, doc_store = load_index()

        assert index is None  # Car le fichier n'existe pas en test
        assert isinstance(doc_store, dict)
