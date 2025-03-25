"""Tests pour le point d'entrée principal de l'application."""

import asyncio
import numpy as np
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import app, lifespan
from app.services import faiss_service
from app.services.embedding_service import EmbeddingService, EXPECTED_DIMENSION

# Client de test existant
client = TestClient(app)

@pytest.fixture
def mock_embedding_service():
    """Fixture pour mocker complètement le service d'embedding.

    Règle appliquée: Testing
    """
    mock_instance = MagicMock()
    # Créer un embedding factice de la bonne dimension
    mock_embedding = np.zeros(EXPECTED_DIMENSION, dtype=np.float32)
    mock_instance.model.encode.return_value = mock_embedding
    mock_instance.encode.return_value = mock_embedding.tolist()

    with patch.object(EmbeddingService, 'get_instance') as mock_get_instance:
        mock_get_instance.return_value = mock_instance
        yield mock_instance

def test_health_endpoint():
    """Test de l'endpoint /health."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_cors_configuration():
    """Test de la configuration CORS."""
    response = client.options("/health", headers={
        "Origin": "http://localhost:3000",
        "Access-Control-Request-Method": "GET",
        "Access-Control-Request-Headers": "Content-Type",
    })
    assert response.status_code == 200
    assert "Access-Control-Allow-Origin" in response.headers
    assert "Access-Control-Allow-Methods" in response.headers

@pytest.mark.asyncio
async def test_lifespan_normal_startup(mock_embedding_service):
    """Test du cycle de vie normal de l'application."""
    app_mock = FastAPI()

    with patch.object(faiss_service, 'load_index') as mock_load_index:
        # Le mock_embedding_service est déjà injecté via la fixture
        async with lifespan(app_mock):
            # Vérifier que les services sont initialisés
            assert mock_load_index.call_count >= 1
            # Pas besoin de vérifier l'appel à get_instance car c'est géré par la fixture

@pytest.mark.asyncio
async def test_lifespan_startup_failure():
    """Test de l'échec d'initialisation des services."""
    app_mock = FastAPI()

    with patch.object(faiss_service, 'load_index') as mock_load_index, \
         pytest.raises(Exception):

        mock_load_index.side_effect = Exception("Erreur simulée")
        async with lifespan(app_mock):
            pass  # Ne devrait pas être exécuté

def test_app_metadata():
    """Test des métadonnées de l'application."""
    assert app.title is not None
    assert app.version is not None
    assert app.description is not None

@pytest.mark.asyncio
async def test_periodic_update(mock_embedding_service):
    """Test du service de mise à jour périodique."""
    with patch.object(faiss_service, 'update_periodically') as mock_update:
        mock_update.return_value = None

        # Simuler le démarrage de l'application
        app_mock = FastAPI()
        with patch.object(faiss_service, 'load_index'):
            async with lifespan(app_mock):
                # Attendre un court instant pour que le thread démarre
                await asyncio.sleep(0.1)
                # Vérifier que update_periodically a été appelé
                assert mock_update.called
