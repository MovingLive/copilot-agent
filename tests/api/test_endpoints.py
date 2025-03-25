"""
Tests unitaires pour les endpoints API.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

def test_root_endpoint():
    """Test de l'endpoint racine."""
    response = client.get("/api")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API Copilot LLM!"}

def test_health_check():
    """Test de l'endpoint de santé."""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_copilot_query_missing_token():
    """Test de l'endpoint Copilot sans token d'authentification."""
    response = client.post(
        "/api",
        json={"messages": [{"role": "user", "content": "test"}]}
    )
    assert response.status_code == 401
    assert "Token d'authentification manquant" in response.json()["detail"]

@pytest.mark.asyncio
async def test_copilot_query_invalid_request():
    """Test de l'endpoint Copilot avec une requête invalide."""
    response = client.post(
        "/api",
        headers={"x-github-token": "fake_token"},
        json={"messages": []}  # Messages vides
    )
    assert response.status_code == 400
    assert "Messages manquants dans la requête" in response.json()["detail"]

@pytest.mark.asyncio
async def test_copilot_query_success():
    """Test d'une requête Copilot réussie."""
    # Mock pour get_github_user
    mock_github_user = AsyncMock(return_value="test_user")
    # Mock pour retrieve_similar_documents
    mock_retrieve_docs = MagicMock(return_value=[
        {
            "content": "Document de test",
            "distance": 0.5,
            "metadata": {"file_path": "test.md"}
        }
    ])

    # Configuration du mock streaming
    mock_response = AsyncMock()
    mock_response.aiter_bytes.return_value = [b'{"choices":[{"message":{"content":"Test response"}}]}']
    mock_response.__aenter__.return_value = mock_response

    mock_client = AsyncMock()
    mock_client.stream.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client

    with (patch("app.api.copilot.get_github_user", mock_github_user),
          patch("app.api.copilot.retrieve_similar_documents", mock_retrieve_docs),
          patch("httpx.AsyncClient", return_value=mock_client)):

        response = client.post(
            "/api",
            headers={"x-github-token": "fake_token"},
            json={
                "messages": [{"role": "user", "content": "test question"}],
                "copilot_references": "contexte additionnel"
            }
        )

        assert response.status_code == 200
        content = b"".join([chunk for chunk in response.streaming_content])
        assert b'{"choices":[{"message":{"content":"Test response"}}]}' in content

@pytest.mark.asyncio
async def test_copilot_query_with_context():
    """Test de l'endpoint Copilot avec contexte additionnel."""
    # Mock pour get_github_user
    mock_github_user = AsyncMock(return_value="test_user")
    # Mock pour retrieve_similar_documents avec résultats détaillés
    mock_retrieve_docs = MagicMock(return_value=[
        {
            "content": "Premier document de test",
            "distance": 0.3,
            "metadata": {
                "file_path": "test1.md",
                "segment_index": 0
            }
        },
        {
            "content": "Deuxième document de test",
            "distance": 0.5,
            "metadata": {
                "file_path": "test2.md",
                "segment_index": 1
            }
        }
    ])

    # Configuration du mock streaming
    mock_response = AsyncMock()
    mock_response.aiter_bytes.return_value = [b'{"choices":[{"message":{"content":"Response with context"}}]}']
    mock_response.__aenter__.return_value = mock_response

    mock_client = AsyncMock()
    mock_client.stream.return_value = mock_response
    mock_client.__aenter__.return_value = mock_client

    with (patch("app.api.copilot.get_github_user", mock_github_user),
          patch("app.api.copilot.retrieve_similar_documents", mock_retrieve_docs),
          patch("httpx.AsyncClient", return_value=mock_client)):

        response = client.post(
            "/api",
            headers={"x-github-token": "fake_token"},
            json={
                "messages": [{"role": "user", "content": "test"}],
                "copilot_references": "contexte spécifique"
            }
        )

        assert response.status_code == 200
        content = b"".join([chunk for chunk in response.streaming_content])
        assert b'{"choices":[{"message":{"content":"Response with context"}}]}' in content

        # Vérifier que retrieve_similar_documents a été appelé avec le bon contexte
        mock_retrieve_docs.assert_called_once_with(
            "test contexte spécifique",
            k=5
        )