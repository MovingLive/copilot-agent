"""Tests unitaires du service FAISS."""

import numpy as np
import pytest

from app.services.faiss_service import (
    load_index,
    retrieve_similar_documents,
    update_periodically,
    _process_search_results,
    _prepare_query_vector,
    _state,
)

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

@pytest.fixture(autouse=True)
def setup_faiss_state(mock_index, document_store):
    """Configure l'état du service FAISS pour les tests."""
    _state.index = mock_index
    _state.document_store = document_store
    yield
    _state.index = None
    _state.document_store = {}

def test_search_similar_success(mocker):
    """Teste la recherche de documents similaires avec succès."""
    mocker.patch(
        "app.services.embedding_service.generate_query_vector",
        return_value=np.zeros((1, MOCK_DIMENSION))
    )

    results = retrieve_similar_documents(TEST_QUERY, TEST_K)

    assert len(results) == TEST_K
    for result in results:
        assert "content" in result
        assert "distance" in result
        assert "metadata" in result

def test_search_similar_no_index(mocker):
    """Teste la recherche quand l'index n'est pas initialisé."""
    _state.index = None
    mocker.patch(
        "app.services.faiss_service.load_index",
        return_value=(None, {})
    )

    results = retrieve_similar_documents(TEST_QUERY)
    assert len(results) == 0

def test_search_similar_invalid_query(mocker):
    """Teste la recherche avec une requête invalide."""
    mocker.patch(
        "app.services.embedding_service.generate_query_vector",
        side_effect=ValueError("Invalid query")
    )

    results = retrieve_similar_documents("")
    assert len(results) == 0

def test_search_similar_faiss_error(mock_index, mocker):
    """Teste la gestion des erreurs FAISS."""
    mocker.patch(
        "app.services.embedding_service.generate_query_vector",
        return_value=np.zeros((1, MOCK_DIMENSION))
    )
    mock_index.search.side_effect = RuntimeError("FAISS error")

    results = retrieve_similar_documents(TEST_QUERY)
    assert len(results) == 0

@pytest.mark.asyncio
async def test_periodic_update(mocker):
    """Teste la mise à jour périodique de l'index."""
    mock_sleep = mocker.patch("asyncio.sleep", side_effect=InterruptedError)
    mock_load = mocker.patch("app.services.faiss_service.load_index")

    with pytest.raises(InterruptedError):
        await update_periodically()

    mock_load.assert_called_once()
    mock_sleep.assert_called_once_with(3600)

def test_process_results_empty_index():
    """Teste le traitement des résultats avec un index vide."""
    _state.index = None
    distances = np.array([[0.1, 0.2]])
    indices = np.array([[-1, -1]])  # Indices invalides

    results = _process_search_results(distances, indices)
    assert len(results) == 0

def test_process_results_invalid_document():
    """Teste le traitement des résultats avec un document invalide."""
    _state.document_store = {"3": {"invalid": "document"}}
    distances = np.array([[0.1]])
    indices = np.array([[3]])

    results = _process_search_results(distances, indices)
    assert len(results) == 0

@pytest.mark.integration
def test_load_index_integration(tmp_path, mocker):
    """Test d'intégration du chargement de l'index."""
    mocker.patch(
        "app.services.faiss_service._get_local_path",
        return_value=str(tmp_path / "nonexistent.faiss")
    )
    mocker.patch(
        "app.services.faiss_service.is_local_environment",
        return_value=True
    )

    index, doc_store = load_index()
    assert index is None
    assert isinstance(doc_store, dict)

def test_prepare_query_vector_mismatch_dimension():
    """Teste la préparation d'un vecteur avec une dimension différente."""
    query_vector = np.zeros((1, MOCK_DIMENSION + 10))  # Dimension plus grande
    prepared = _prepare_query_vector(query_vector)

    assert prepared.shape == (1, MOCK_DIMENSION)
    assert np.all(prepared[:, :MOCK_DIMENSION] == 0)