"""Tests unitaires du service FAISS."""

import numpy as np
import pytest
from fastapi import HTTPException

from app.services.faiss_service import FAISSService

# Constantes de test
TEST_QUERY = "Test query"
TEST_K = 3
MOCK_DIMENSION = 384  # Dimension du modèle all-MiniLM-L6-v2

@pytest.fixture(name="mock_index")
def fixture_mock_index(mocker):
    """Crée un mock d'un index FAISS."""
    mock = mocker.Mock()
    mock.d = MOCK_DIMENSION
    mock.ntotal = 10
    mock.search.return_value = (
        np.array([[0.1, 0.2, 0.3]]),  # distances
        np.array([[0, 1, 2]])  # indices
    )
    return mock

@pytest.fixture(name="document_store")
def fixture_document_store():
    """Crée un store de documents de test."""
    return {
        "0": {"content": "Document 0", "source": "test0.py"},
        "1": {"content": "Document 1", "source": "test1.py"},
        "2": {"content": "Document 2", "source": "test2.py"}
    }

@pytest.fixture(name="faiss_service")
def fixture_faiss_service(mock_index, document_store):
    """Crée une instance de test du service FAISS."""
    service = FAISSService()
    service.index = mock_index
    service.document_store = document_store
    return service

def test_singleton_instance():
    """Vérifie que le service FAISS est bien un singleton."""
    service1 = FAISSService.get_instance()
    service2 = FAISSService.get_instance()
    assert service1 is service2

def test_search_similar_success(faiss_service, mocker):
    """Teste la recherche de documents similaires avec succès."""
    mocker.patch(
        "app.services.faiss_service.generate_query_vector",
        return_value=np.zeros((1, MOCK_DIMENSION))
    )

    results = faiss_service.search_similar(TEST_QUERY, TEST_K)

    assert len(results) == TEST_K
    for result in results:
        assert "content" in result
        assert "distance" in result
        assert "metadata" in result
        assert "source" in result["metadata"]

def test_search_similar_no_index(faiss_service):
    """Teste la recherche quand l'index n'est pas initialisé."""
    faiss_service.index = None

    with pytest.raises(HTTPException) as exc_info:
        faiss_service.search_similar(TEST_QUERY)

    assert exc_info.value.status_code == 500
    assert "Index FAISS non initialisé" in str(exc_info.value.detail)

def test_search_similar_invalid_query(faiss_service, mocker):
    """Teste la recherche avec une requête invalide."""
    mocker.patch(
        "app.services.faiss_service.generate_query_vector",
        side_effect=ValueError("Invalid query")
    )

    with pytest.raises(HTTPException) as exc_info:
        faiss_service.search_similar("")

    assert exc_info.value.status_code == 400

def test_search_similar_faiss_error(faiss_service, mocker):
    """Teste la gestion des erreurs FAISS."""
    mocker.patch(
        "app.services.faiss_service.generate_query_vector",
        return_value=np.zeros((1, MOCK_DIMENSION))
    )
    faiss_service.index.search.side_effect = RuntimeError("FAISS error")

    with pytest.raises(HTTPException) as exc_info:
        faiss_service.search_similar(TEST_QUERY)

    assert exc_info.value.status_code == 500
    assert "Erreur lors de la recherche FAISS" in str(exc_info.value.detail)

@pytest.mark.asyncio
async def test_periodic_update(faiss_service, mocker):
    """Teste la mise à jour périodique de l'index."""
    mock_sleep = mocker.patch("time.sleep", side_effect=InterruptedError)
    mock_load = mocker.patch.object(faiss_service, "load_index")

    with pytest.raises(InterruptedError):
        await faiss_service.update_periodically()

    mock_load.assert_called_once()
    mock_sleep.assert_called_once_with(3600)

def test_process_results_empty_index(faiss_service):
    """Teste le traitement des résultats avec un index vide."""
    distances = np.array([[0.1, 0.2]])
    indices = np.array([[-1, -1]])  # Indices invalides

    results = faiss_service.process_search_results(distances, indices)

    assert len(results) == 0

def test_process_results_invalid_document(faiss_service):
    """Teste le traitement des résultats avec un document invalide."""
    faiss_service.document_store["3"] = {"invalid": "document"}
    distances = np.array([[0.1]])
    indices = np.array([[3]])

    results = faiss_service.process_search_results(distances, indices)

    assert len(results) == 0

@pytest.mark.integration
def test_load_index_integration(tmp_path, mocker):
    """Test d'intégration du chargement de l'index."""
    mocker.patch("app.core.config.settings.ENV", "local")
    mocker.patch(
        "app.core.config.settings.LOCAL_OUTPUT_DIR",
        return_value=str(tmp_path)
    )

    service = FAISSService()

    with pytest.raises(HTTPException) as exc_info:
        service.load_index()

    assert exc_info.value.status_code == 500
    assert "Erreur lors du chargement des fichiers" in str(exc_info.value.detail)