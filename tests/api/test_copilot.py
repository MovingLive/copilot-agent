"""Tests pour les endpoints Copilot."""
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app

client = TestClient(app)

# Données de test
TEST_TOKEN = "fake_github_token"
TEST_USER = "testuser"
TEST_QUERY = "Comment utiliser FastAPI?"
TEST_CONTEXT = "Documentation FastAPI"


@pytest.mark.asyncio
async def test_handle_copilot_query_success():
    """Test du succès de la requête Copilot."""
    headers = {"x-github-token": TEST_TOKEN}
    data = {
        "messages": [{"role": "user", "content": TEST_QUERY}],
        "copilot_references": TEST_CONTEXT,
    }

    async def mock_get_github_user(token):
        return TEST_USER

    async def mock_generate_streaming_response(req_data, token):
        yield b'{"response": "test"}'

    with (
        patch("app.api.copilot.get_github_user", side_effect=mock_get_github_user),
        patch("app.services.faiss_service.retrieve_similar_documents", return_value=[
            {"content": "FastAPI est un framework moderne pour Python"}
        ]),
        patch("app.api.copilot.generate_streaming_response", 
              side_effect=mock_generate_streaming_response),
    ):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/", json=data, headers=headers)
            assert response.status_code == 200


def test_handle_copilot_query_missing_token():
    """Test de l'erreur lors d'un token manquant."""
    response = client.post("/", json={"messages": [{"content": TEST_QUERY}]})
    assert response.status_code == 401
    assert response.json()["detail"] == "Token d'authentification manquant"


def test_handle_copilot_query_missing_messages():
    """Test de l'erreur lors de messages manquants."""
    headers = {"x-github-token": TEST_TOKEN}
    response = client.post("/", json={}, headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"] == "Messages manquants dans la requête"


def test_handle_copilot_query_empty_message():
    """Test de l'erreur lors d'un message vide."""
    headers = {"x-github-token": TEST_TOKEN}
    data = {"messages": [{"role": "user", "content": ""}]}
    response = client.post("/", json=data, headers=headers)
    assert response.status_code == 400
    assert response.json()["detail"] == "Message vide"


@pytest.mark.asyncio
async def test_handle_copilot_query_with_context():
    """Test de la requête avec contexte additionnel."""
    headers = {"x-github-token": TEST_TOKEN}
    data = {
        "messages": [{"role": "user", "content": TEST_QUERY}],
        "copilot_references": TEST_CONTEXT,
    }

    async def mock_get_github_user(token):
        return TEST_USER

    async def mock_generate_streaming_response(req_data, token):
        yield b'{"response": "test with context"}'

    with (
        patch("app.api.copilot.get_github_user", side_effect=mock_get_github_user),
        patch("app.services.faiss_service.retrieve_similar_documents", return_value=[
            {"content": "FastAPI est un framework moderne"},
            {"content": "Exemple d'utilisation de FastAPI"}
        ]),
        patch("app.api.copilot.generate_streaming_response", 
              side_effect=mock_generate_streaming_response),
    ):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/", json=data, headers=headers)
            assert response.status_code == 200


@pytest.mark.asyncio
async def test_handle_copilot_query_service_error():
    """Test de la gestion des erreurs de service."""
    headers = {"x-github-token": TEST_TOKEN}
    data = {"messages": [{"role": "user", "content": TEST_QUERY}]}

    async def mock_get_github_user(token):
        raise HTTPException(status_code=401, detail="Token GitHub invalide")

    with patch("app.api.copilot.get_github_user", side_effect=mock_get_github_user):
        async with AsyncClient(app=app, base_url="http://test") as ac:
            response = await ac.post("/", json=data, headers=headers)
            assert response.status_code == 401
            assert response.json()["detail"] == "Token GitHub invalide"