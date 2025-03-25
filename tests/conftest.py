"""Configuration et fixtures partagées pour les tests.

Ce module contient les fixtures pytest réutilisables dans tous les tests.
"""

import asyncio
from typing import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.core.config import settings
from app.main import app
from app.services import faiss_service, embedding_service
from app.services.embedding_service import EmbeddingService

# Configuration pour les tests
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Crée une boucle d'événements pour les tests asynchrones."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="session")
def test_client() -> Generator[TestClient, None, None]:
    """Fournit un client HTTP synchrone pour les tests."""
    with TestClient(app) as client:
        yield client

@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """Fournit un client HTTP asynchrone pour les tests."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client

@pytest.fixture(scope="session", autouse=True)
def initialize_services():
    """Initialise les services nécessaires pour les tests."""
    # Configuration du modèle d'embedding
    _ = EmbeddingService.get_instance().model

    # Chargement de l'index FAISS
    index, doc_store = faiss_service.load_index()
    faiss_service._state.index = index
    faiss_service._state.document_store = doc_store

    yield

    # Nettoyage après les tests
    faiss_service._state.index = None
    faiss_service._state.document_store = {}

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Configure les variables d'environnement pour les tests."""
    monkeypatch.setenv("ENV", "test")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("S3_BUCKET_NAME", "test-bucket")

    yield

@pytest.fixture
def mock_faiss_service(mocker):
    """Mock du service FAISS pour les tests."""
    mock_search = mocker.patch("app.services.faiss_service.retrieve_similar_documents")
    mock_search.return_value = [
        {
            "content": "Test content",
            "distance": 0.5,
            "metadata": {"source": "test.py"}
        }
    ]
    return mock_search

@pytest.fixture
def mock_embedding_service(mocker):
    """Mock du service d'embeddings pour les tests."""
    mock_generate = mocker.patch("app.services.embedding_service.generate_query_vector")
    mock_generate.return_value = mocker.Mock()
    return mock_generate

@pytest.fixture
def mock_copilot_response():
    """Mock d'une réponse Copilot pour les tests."""
    return {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": "Voici une réponse de test"
                }
            }
        ]
    }

# Configuration des marqueurs de test
def pytest_configure(config):
    """Configure les marqueurs de test personnalisés."""
    config.addinivalue_line(
        "markers",
        "slow: marque les tests lents qui peuvent être ignorés avec -m 'not slow'"
    )
    config.addinivalue_line(
        "markers",
        "integration: marque les tests d'intégration"
    )
    config.addinivalue_line(
        "markers",
        "e2e: marque les tests end-to-end"
    )