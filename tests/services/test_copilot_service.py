"""
Tests unitaires pour le service Copilot.
"""

import pytest
from fastapi import HTTPException
import httpx
import pytest
from unittest.mock import AsyncMock, patch

from app.services.copilot_service import (
    get_github_user,
    format_copilot_messages,
    call_copilot_api
)

@pytest.mark.asyncio
async def test_get_github_user_success():
    """Test de la récupération réussie des informations utilisateur."""
    mock_response = AsyncMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"login": "test_user"}

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        user_login = await get_github_user("fake_token")
        assert user_login == "test_user"

@pytest.mark.asyncio
async def test_get_github_user_unauthorized():
    """Test avec un token invalide."""
    mock_response = AsyncMock()
    mock_response.status_code = 401
    mock_response.raise_for_status.side_effect = httpx.HTTPError("Unauthorized")

    with patch("httpx.AsyncClient.get", return_value=mock_response):
        with pytest.raises(HTTPException) as excinfo:
            await get_github_user("invalid_token")
        assert excinfo.value.status_code == 401
        assert "Token GitHub invalide" in str(excinfo.value.detail)

def test_format_copilot_messages():
    """Test du formatage des messages pour l'API Copilot."""
    query = "Comment créer une GitHub Action?"
    context = "Documentation sur les GitHub Actions"
    user_login = "test_user"

    messages = format_copilot_messages(query, context, user_login)

    assert isinstance(messages, list)
    assert len(messages) >= 4  # Au moins 4 messages (3 system + 1 user)
    assert all(isinstance(msg, dict) for msg in messages)
    assert all("role" in msg and "content" in msg for msg in messages)

    # Vérification des messages système
    system_messages = [msg for msg in messages if msg["role"] == "system"]
    assert len(system_messages) >= 3
    assert any("@test_user" in msg["content"] for msg in system_messages)
    assert any(context in msg["content"] for msg in system_messages)

    # Vérification du message utilisateur
    user_messages = [msg for msg in messages if msg["role"] == "user"]
    assert len(user_messages) == 1
    assert user_messages[0]["content"] == query

@pytest.mark.asyncio
async def test_call_copilot_api_success():
    """Test d'un appel réussi à l'API Copilot."""
    messages = [{"role": "user", "content": "test"}]
    mock_response = AsyncMock()
    mock_response.json.return_value = {
        "choices": [{"message": {"content": "Réponse de test"}}]
    }
    mock_response.raise_for_status = AsyncMock()
    mock_response.is_success = True

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        response = await call_copilot_api(messages, "fake_token")
        assert response == "Réponse de test"

@pytest.mark.asyncio
async def test_call_copilot_api_error():
    """Test d'un appel échoué à l'API Copilot."""
    messages = [{"role": "user", "content": "test"}]
    mock_response = AsyncMock()
    mock_response.status_code = 500
    mock_response.raise_for_status.side_effect = httpx.HTTPError("Server error")
    mock_response.is_success = False

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        with pytest.raises(HTTPException) as excinfo:
            await call_copilot_api(messages, "fake_token")
        assert excinfo.value.status_code == 500

@pytest.mark.asyncio
async def test_call_copilot_api_invalid_response():
    """Test avec une réponse invalide de l'API Copilot."""
    messages = [{"role": "user", "content": "test"}]
    mock_response = AsyncMock()
    mock_response.json.return_value = {"invalid": "format"}
    mock_response.raise_for_status = AsyncMock()
    mock_response.is_success = True

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        with pytest.raises(ValueError) as excinfo:
            await call_copilot_api(messages, "fake_token")
        assert "Format de réponse inattendu" in str(excinfo.value)