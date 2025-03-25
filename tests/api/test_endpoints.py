"""
Tests unitaires pour les endpoints API.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.copilot_service import get_github_user
from app.services.faiss_service import retrieve_similar_documents

client = TestClient(app)

def test_root_endpoint():
    """Test de l'endpoint racine."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API Copilot LLM!"}

def test_health_check():
    """Test de l'endpoint de santé."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_copilot_query_missing_token():
    """Test de l'endpoint Copilot sans token d'authentification."""
    response = client.post(
        "/",
        json={"messages": [{"role": "user", "content": "test"}]}
    )
    assert response.status_code == 401
    assert "Missing authentication token" in response.json()["detail"]

@pytest.mark.asyncio
async def test_copilot_query_invalid_request():
    """Test de l'endpoint Copilot avec une requête invalide."""
    response = client.post(
        "/",
        headers={"x-github-token": "fake_token"},
        json={"messages": []}  # Messages vides
    )
    assert response.status_code == 400

@pytest.mark.asyncio
async def test_copilot_query_success():
    """Test d'une requête Copilot réussie."""
    # Mock pour get_github_user
    mock_github_user = AsyncMock(return_value="test_user")
    # Mock pour retrieve_similar_documents
    mock_retrieve_docs = AsyncMock(return_value=[
        {"content": "Document de test"}
    ])

    # Mock pour la réponse de l'API Copilot en streaming
    async def mock_stream(*args, **kwargs):
        yield b'{"choices":[{"message":{"content":"Réponse test"}}]}'

    with patch("app.api.copilot.get_github_user", mock_github_user), \
         patch("app.api.copilot.retrieve_similar_documents", mock_retrieve_docs), \
         patch("httpx.AsyncClient.stream", new_callable=AsyncMock) as mock_stream:

        mock_stream.return_value.__aenter__.return_value.aiter_bytes = mock_stream
        mock_stream.return_value = mock_stream

        response = client.post(
            "/",
            headers={"x-github-token": "fake_token"},
            json={
                "messages": [{"role": "user", "content": "test question"}],
                "copilot_references": "contexte additionnel"
            }
        )

        assert response.status_code == 200
        content = json.loads(response.content)
        assert "choices" in content
        assert "message" in content["choices"][0]
        assert content["choices"][0]["message"]["content"] == "Réponse test"

@pytest.mark.asyncio
async def test_copilot_query_with_context():
    """Test de l'endpoint Copilot avec contexte additionnel."""
    mock_github_user = AsyncMock(return_value="test_user")
    mock_retrieve_docs = AsyncMock(return_value=[
        {"content": "Premier document"},
        {"content": "Deuxième document"}
    ])

    async def mock_stream(*args, **kwargs):
        yield b'{"choices":[{"message":{"content":"Réponse avec contexte"}}]}'

    with patch("app.api.copilot.get_github_user", mock_github_user), \
         patch("app.api.copilot.retrieve_similar_documents", mock_retrieve_docs), \
         patch("httpx.AsyncClient.stream", new_callable=AsyncMock) as mock_stream:

        mock_stream.return_value.__aenter__.return_value.aiter_bytes = mock_stream
        mock_stream.return_value = mock_stream

        response = client.post(
            "/",
            headers={"x-github-token": "fake_token"},
            json={
                "messages": [{"role": "user", "content": "test"}],
                "copilot_references": "contexte spécifique"
            }
        )

        assert response.status_code == 200
        # Vérifier que retrieve_similar_documents a été appelé avec le bon contexte
        mock_retrieve_docs.assert_called_once_with(
            "test contexte spécifique",
            k=5
        )