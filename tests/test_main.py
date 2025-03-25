"""Tests pour le point d'entrée principal de l'application."""

import asyncio
from unittest.mock import patch, MagicMock
import pytest
from fastapi.testclient import TestClient
from fastapi import FastAPI

from app.main import app, lifespan
from app.services import faiss_service
from app.services.embedding_service import EmbeddingService

# Client de test existant
client = TestClient(app)

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
async def test_lifespan_normal_startup():
    """Test du cycle de vie normal de l'application."""
    app_mock = FastAPI()

    with patch.object(faiss_service, 'load_index'), \
         patch.object(EmbeddingService, 'get_instance') as mock_embedding:

        mock_embedding.return_value = MagicMock()
        async with lifespan(app_mock):
            # Vérifier que les services sont initialisés
            faiss_service.load_index.assert_called_once()
            mock_embedding.assert_called_once()

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
async def test_periodic_update():
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
