"""Tests E2E de l'API Copilot.

Ces tests vérifient le comportement de l'API dans son ensemble.
"""

import asyncio
import json
from typing import AsyncGenerator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.core.config import settings
from app.main import app
from app.services import faiss_service

# Configuration des tests
MOCK_GITHUB_TOKEN = "mock_token"
MOCK_QUERY = "Comment implémenter un pattern Observer?"
TEST_K = 3

@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Fixture fournissant un client HTTP asynchrone."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture
def test_client() -> TestClient:
    """Fixture fournissant un client HTTP synchrone."""
    return TestClient(app)

@pytest.fixture(autouse=True)
def setup_faiss():
    """Fixture initialisant l'index FAISS pour les tests."""
    faiss_service.load_faiss_index()
    yield
    # Nettoyage après les tests si nécessaire

def test_health_check(test_client: TestClient):
    """Test de l'endpoint de santé."""
    response = test_client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

@pytest.mark.asyncio
async def test_root_endpoint(async_client: AsyncClient):
    """Test de l'endpoint racine."""
    response = await async_client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Bienvenue dans l'API Copilot LLM!"
    }

@pytest.mark.asyncio
async def test_copilot_query_without_token(async_client: AsyncClient):
    """Test d'une requête Copilot sans token."""
    response = await async_client.post(
        "/api/",
        json={
            "messages": [{"role": "user", "content": MOCK_QUERY}]
        }
    )
    assert response.status_code == 401
    assert "Missing authentication token" in response.json()["detail"]

@pytest.mark.asyncio
async def test_copilot_query_with_invalid_token(async_client: AsyncClient):
    """Test d'une requête Copilot avec un token invalide."""
    response = await async_client.post(
        "/api/",
        headers={"x-github-token": "invalid_token"},
        json={
            "messages": [{"role": "user", "content": MOCK_QUERY}]
        }
    )
    assert response.status_code == 401
    assert "Invalid GitHub token" in response.json()["detail"]

@pytest.mark.asyncio
async def test_copilot_query_with_context(async_client: AsyncClient):
    """Test d'une requête Copilot avec contexte."""
    response = await async_client.post(
        "/api/",
        headers={"x-github-token": MOCK_GITHUB_TOKEN},
        json={
            "messages": [{"role": "user", "content": MOCK_QUERY}],
            "copilot_references": "contexte additionnel"
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "choices" in result
    assert len(result["choices"]) > 0
    assert "message" in result["choices"][0]
    assert "content" in result["choices"][0]["message"]

@pytest.mark.asyncio
async def test_similar_documents_search(async_client: AsyncClient):
    """Test de la recherche de documents similaires."""
    response = await async_client.post(
        "/api/search",
        headers={"x-github-token": MOCK_GITHUB_TOKEN},
        json={
            "query": MOCK_QUERY,
            "k": TEST_K
        }
    )
    assert response.status_code == 200
    results = response.json()
    assert isinstance(results, list)
    assert len(results) <= TEST_K
    for doc in results:
        assert "content" in doc
        assert "distance" in doc
        assert isinstance(doc["distance"], float)

@pytest.mark.asyncio
async def test_concurrent_requests(async_client: AsyncClient):
    """Test de multiples requêtes concurrentes."""
    async def make_request():
        return await async_client.post(
            "/api/",
            headers={"x-github-token": MOCK_GITHUB_TOKEN},
            json={
                "messages": [{"role": "user", "content": MOCK_QUERY}]
            }
        )

    # Exécution de 5 requêtes en parallèle
    responses = await asyncio.gather(
        *[make_request() for _ in range(5)]
    )

    for response in responses:
        assert response.status_code == 200
        result = response.json()
        assert "choices" in result
        assert len(result["choices"]) > 0

@pytest.mark.asyncio
async def test_invalid_request_body(async_client: AsyncClient):
    """Test avec un corps de requête invalide."""
    response = await async_client.post(
        "/api/",
        headers={"x-github-token": MOCK_GITHUB_TOKEN},
        json={"invalid": "request"}
    )
    assert response.status_code == 422  # Validation error

@pytest.mark.asyncio
async def test_large_query(async_client: AsyncClient):
    """Test avec une requête très longue."""
    long_query = "test " * 1000
    response = await async_client.post(
        "/api/",
        headers={"x-github-token": MOCK_GITHUB_TOKEN},
        json={
            "messages": [{"role": "user", "content": long_query}]
        }
    )
    assert response.status_code == 200