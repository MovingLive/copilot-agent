"""Tests unitaires pour le service d'embeddings."""

import numpy as np
import pytest
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from app.services.embedding_service import (
    EmbeddingService,
    embed_text,
    generate_query_vector,
    normalize_vector
)

# Constantes de test
TEST_TEXT = "Test text"
MOCK_DIMENSION = 384
MOCK_VECTOR = np.random.rand(1, MOCK_DIMENSION)

@pytest.fixture(name="mock_transformer")
def fixture_mock_transformer(mocker):
    """Crée un mock du modèle SentenceTransformer."""
    mock = mocker.Mock(spec=SentenceTransformer)
    mock.encode.return_value = MOCK_VECTOR
    mock.get_sentence_embedding_dimension.return_value = MOCK_DIMENSION
    return mock

@pytest.fixture(name="embedding_service")
def fixture_embedding_service(mock_transformer):
    """Configure le service d'embedding pour les tests."""
    service = EmbeddingService()
    service.model = mock_transformer  # Utilisation de la property
    return service

def test_singleton_pattern():
    """Vérifie que le pattern singleton fonctionne."""
    instance1 = EmbeddingService.get_instance()
    instance2 = EmbeddingService.get_instance()
    assert instance1 is instance2

def test_model_initialization(mock_transformer, mocker):
    """Vérifie l'initialisation du modèle."""
    mocker.patch(
        "app.services.embedding_service.SentenceTransformer",
        return_value=mock_transformer
    )

    service = EmbeddingService()
    model = service.model  # Utilisation de la property

    assert model is mock_transformer
    assert model.get_sentence_embedding_dimension() == MOCK_DIMENSION

def test_generate_vector_success(mock_transformer):
    """Vérifie la génération réussie d'un vecteur."""
    vector = generate_query_vector(TEST_TEXT)

    assert isinstance(vector, np.ndarray)
    assert vector.shape == (1, MOCK_DIMENSION)
    mock_transformer.encode.assert_called_once()

def test_empty_input_validation():
    """Vérifie la validation des entrées vides."""
    with pytest.raises(HTTPException) as exc_info:
        generate_query_vector("")
    assert exc_info.value.status_code == 400

def test_model_error(mock_transformer):
    """Vérifie la gestion des erreurs du modèle."""
    mock_transformer.encode.side_effect = RuntimeError("Test error")

    with pytest.raises(HTTPException) as exc_info:
        generate_query_vector(TEST_TEXT)
    assert exc_info.value.status_code == 500

def test_vector_normalization():
    """Vérifie la normalisation des vecteurs."""
    vector = np.array([[1.0, 2.0, 3.0]])
    normalized = normalize_vector(vector)

    assert np.allclose(np.linalg.norm(normalized), 1.0)

def test_embedding_text_success(mock_transformer):
    """Vérifie l'embedding réussi d'un texte."""
    result = embed_text(TEST_TEXT)

    assert isinstance(result, list)
    assert len(result) == MOCK_DIMENSION
    mock_transformer.encode.assert_called_once()