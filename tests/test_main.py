import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    # On suppose que l'endpoint /health retourne {"status": "ok"}
    assert response.json().get("status") == "ok"
