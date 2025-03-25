import numpy as np
import pytest
import torch
from fastapi import HTTPException
from app.services import embedding_service

# ...existing imports...

# Fake classes pour simuler le modèle et le tensor
class FakeTensor:
    def __init__(self, data):
        self.data = data
    def cpu(self):
        return self
    def numpy(self):
        return self.data

class FakeModel:
    def encode(self, x, convert_to_tensor=True, normalize_embeddings=True):
        if isinstance(x, list):
            # Retourne un tenseur de dimension (len(x), EXPECTED_DIMENSION)
            data = np.ones((len(x), embedding_service.EXPECTED_DIMENSION))
        else:
            # Retourne un tenseur de dimension (1, EXPECTED_DIMENSION)
            data = np.ones((1, embedding_service.EXPECTED_DIMENSION))
            data = data[0]  # Retourne un vecteur 1D de dimension EXPECTED_DIMENSION
        return FakeTensor(data)

# Fixture pour installer le FakeModel dans l'EmbeddingService
@pytest.fixture(autouse=True)
def fake_model(monkeypatch):
    fake_instance = type("FakeInstance", (), {})()
    fake_instance.model = FakeModel()
    monkeypatch.setattr(embedding_service.EmbeddingService, "get_instance", lambda: fake_instance)

def test_validate_input_valid():
    # ...existing code...
    try:
        embedding_service.validate_input("texte valide")
    except Exception:
        pytest.fail("validate_input a échoué avec un texte valide")

def test_validate_input_invalid():
    with pytest.raises(ValueError):
        embedding_service.validate_input("   ")

def test_normalize_vector():
    vec = np.array([3.0, 4.0])
    normalized = embedding_service.normalize_vector(vec)
    assert np.allclose(np.linalg.norm(normalized), 1.0)

def test_embed_text_success():
    result = embedding_service.embed_text("texte pour embedding")
    # On attend un vecteur planaire de dimension 384
    assert isinstance(result, list)
    assert len(result) in (384, )  # dimension attendue

def test_embed_text_invalid(monkeypatch):
    with pytest.raises(HTTPException) as exc_info:
        embedding_service.embed_text("   ")
    assert exc_info.value.status_code == 400

def test_generate_query_vector_success():
    vector = embedding_service.generate_query_vector("requête de test")
    # On attend un tenseur numpy 2D avec 1 ligne et EXPECTED_DIMENSION colonnes
    assert isinstance(vector, np.ndarray)
    assert vector.ndim == 2
    assert vector.shape[1] == embedding_service.EXPECTED_DIMENSION

def test_generate_query_vector_invalid():
    with pytest.raises(ValueError):
        embedding_service.generate_query_vector("   ")

# ...existing code...
