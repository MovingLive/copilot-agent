"""Tests pour les endpoints de santé."""
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root_endpoint():
    """Test du point d'entrée principal."""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API Copilot LLM!"}


def test_health_check_endpoint():
    """Test de l'endpoint de vérification de santé."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}