"""Configuration et fixtures partagées pour les tests.

Ce module contient les fixtures pytest réutilisables dans tous les tests.
"""

import asyncio
import numpy as np
from typing import AsyncGenerator, Generator
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from app.main import app
from app.services import faiss_service
from app.services.embedding_service import EmbeddingService, EXPECTED_DIMENSION

# Configuration pour les tests
@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Crée une boucle d'événements pour les tests asynchrones."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# Classe Mock pour SentenceTransformer
class MockSentenceTransformer(MagicMock):
    """Mock de SentenceTransformer qui hérite de MagicMock pour supporter toutes les méthodes de mock."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encode = MagicMock()
        self.encode.return_value = np.zeros((1, EXPECTED_DIMENSION), dtype=np.float32)
        self.get_sentence_embedding_dimension = MagicMock(return_value=EXPECTED_DIMENSION)

# Cette fixture sera appliquée à TOUS les tests (autouse=True) pour bloquer les requêtes HF
@pytest.fixture(scope="session", autouse=True)
def block_huggingface_requests():
    """
    Bloque toutes les requêtes vers Hugging Face pour éviter les erreurs SSL.
    Remplace complètement la classe SentenceTransformer par un mock.
    """
    # Remplacer complètement la classe SentenceTransformer par notre mock
    with patch('sentence_transformers.SentenceTransformer', MockSentenceTransformer):
        # Mock des requêtes HTTP à bas niveau pour éviter toute connexion réseau
        with patch('requests.get', return_value=MagicMock(status_code=200)):
            with patch('requests.post', return_value=MagicMock(status_code=200)):
                # Bloquer aussi les requêtes SSL directes
                with patch('ssl.get_server_certificate', return_value="MOCK_CERTIFICATE"):
                    yield

@pytest.fixture(scope="function")
def mock_sentence_transformer():
    """
    Mock du modèle SentenceTransformer pour les tests.
    Fournit un mock complet qui ne tente pas de télécharger ou vérifier le modèle.
    """
    mock_model = MockSentenceTransformer()
    
    # Réinitialisation du singleton pour les tests
    EmbeddingService._instance = None
    EmbeddingService._model = None
    
    # Patch complet de SentenceTransformer pour éviter toute initialisation réelle
    with patch('sentence_transformers.SentenceTransformer', return_value=mock_model):
        # Pré-initialiser le singleton pour éviter une nouvelle instanciation pendant le test
        EmbeddingService._instance = EmbeddingService()
        EmbeddingService._model = mock_model
        yield mock_model

@pytest.fixture(scope="function", autouse=True)
def reset_embedding_service():
    """
    Réinitialise le service d'embedding avant chaque test.
    """
    EmbeddingService._instance = None
    EmbeddingService._model = None
    yield
    EmbeddingService._instance = None
    EmbeddingService._model = None

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
    # Pas besoin de configurer explicitement le modèle d'embedding,
    # car il est déjà mocké par mock_sentence_transformer

    # Chargement de l'index FAISS avec gestion des erreurs
    try:
        index, doc_store = faiss_service.load_index()
        if index is not None:
            faiss_service._state.index = index
            faiss_service._state.document_store = doc_store
        else:
            # Si l'index est None, on initialise avec des valeurs par défaut pour les tests
            faiss_service._state.index = None
            faiss_service._state.document_store = {}
    except Exception as e:
        # En cas d'erreur, on initialise avec des valeurs par défaut
        print(f"Erreur lors du chargement de l'index FAISS pour les tests: {e}")
        faiss_service._state.index = None
        faiss_service._state.document_store = {}

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
    """Mock du service d'embeddings pour les tests individuels."""
    mock_generate = mocker.patch("app.services.embedding_service.generate_query_vector")
    mock_generate.return_value = np.zeros((1, EXPECTED_DIMENSION), dtype=np.float32)
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