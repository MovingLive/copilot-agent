"""Tests pour le service FAISS."""

import asyncio
import numpy as np
import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import faiss
import json
import os
import tempfile
from botocore.exceptions import ClientError
from app.services import faiss_service
from app.core.config import settings
from app.services.faiss_service import (
    FAISSServiceError,
    FAISSLoadError,
    FAISSSyncError,
    create_optimized_index,
    configure_search_parameters,
)

class FakeFaissIndex:
    def __init__(self, d):
        self.d = d
        self.ntotal = 1
    def search(self, query_vector, k):
        distances = np.array([[0.1] * k])
        indices = np.array([[0] + [-1]*(k-1)])
        return distances, indices

class FakeSettings:
    MIN_SEGMENT_LENGTH = 0  # pour ne pas filtrer les documents
    FAISS_INDEX_FILE = "index.faiss"
    FAISS_METADATA_FILE = "metadata.json"
    LOCAL_OUTPUT_DIR = "/tmp/faiss_output"
    AWS_REGION = "us-east-1"
    S3_BUCKET_NAME = "test-bucket"

@pytest.fixture(autouse=True)
def fake_settings(monkeypatch):
    monkeypatch.setattr(faiss_service, "settings", FakeSettings())

@pytest.fixture(autouse=True)
def fake_index_and_store():
    faiss_service._state.index = FakeFaissIndex(d=384)
    faiss_service._state.document_store = {
        "0": {"content": "Document de test", "meta": "data"}
    }
    yield
    faiss_service._state.index = None
    faiss_service._state.document_store = {}

def fake_generate_query_vector(query: str):
    return np.ones((1, 384), dtype="float32")

def fake_search_error(query_vector, k, precision_priority=False):
    raise RuntimeError("Search error simulation")

def fake_load_index():
    return FakeFaissIndex(384), {"0": {"content": "Document from load_index", "meta": "info"}}

def test_retrieve_similar_documents_success(monkeypatch):
    monkeypatch.setattr(faiss_service, "generate_query_vector", fake_generate_query_vector)
    results = faiss_service.retrieve_similar_documents("Test query", k=3)
    assert len(results) == 1
    assert results[0]["content"] == "Document de test"

def test_retrieve_similar_documents_dim_mismatch(monkeypatch):
    fake_index = FakeFaissIndex(d=384)
    faiss_service._state.index = fake_index
    faiss_service._state.document_store = {"0": {"content": "Doc mismatch", "meta": "info"}}
    def fake_query_vector(query: str):
        return np.ones((1, 200), dtype="float32")
    monkeypatch.setattr(faiss_service, "generate_query_vector", fake_query_vector)
    results = faiss_service.retrieve_similar_documents("Dim mismatch", k=3)
    assert len(results) == 1
    assert results[0]["content"] == "Doc mismatch"

def test_retrieve_similar_documents_search_error(monkeypatch):
    monkeypatch.setattr(faiss_service, "generate_query_vector", fake_generate_query_vector)
    monkeypatch.setattr(faiss_service, "_search_in_index", fake_search_error)
    results = faiss_service.retrieve_similar_documents("Error query", k=3)
    assert results == []

def test_retrieve_similar_documents_load_index(monkeypatch):
    faiss_service._state.index = None
    monkeypatch.setattr(faiss_service, "load_index", lambda: fake_load_index())
    monkeypatch.setattr(faiss_service, "generate_query_vector", fake_generate_query_vector)
    results = faiss_service.retrieve_similar_documents("Load index", k=3)
    assert len(results) == 1
    assert "Document from load_index" in results[0]["content"]

def test_save_faiss_index(tmp_path, monkeypatch):
    """
    Teste la sauvegarde de l'index FAISS et du mapping avec des mocks.
    """
    # Créer un index de test
    dimension = 128
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    index.add_with_ids(
        np.random.rand(2, dimension).astype("float32"), np.array([1, 2], dtype=np.int64)
    )

    # Créer un mapping de test
    mapping = {
        "1": {"source": "docs/test1.md", "segment": 0},
        "2": {"source": "docs/test2.md", "segment": 0}
    }

    # Mock des fonctions d'écriture
    mock_write_index = MagicMock()
    mock_open = MagicMock()

    monkeypatch.setattr(faiss, "write_index", mock_write_index)
    monkeypatch.setattr("builtins.open", mock_open)

    # Appeler la fonction save_faiss_index
    faiss_service.save_faiss_index(index, mapping, str(tmp_path))

    # Vérifier que les mocks ont été appelés correctement
    mock_write_index.assert_called_once()
    mock_open.assert_called_once_with(
        os.path.join(str(tmp_path), settings.FAISS_METADATA_FILE), "w", encoding="utf-8"
    )

def test_save_faiss_index_file_permissions(tmp_path):
    """
    Teste la gestion des erreurs de permissions lors de la sauvegarde.
    """
    # Créer un index de test
    dimension = 128
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    mapping = {"1": {"source": "test.md", "segment": 0}}

    with patch("builtins.open", side_effect=PermissionError), pytest.raises(PermissionError):
        faiss_service.save_faiss_index(index, mapping, str(tmp_path))

def test_create_optimized_index_small_dataset():
    """Teste la création d'un index optimisé pour un petit jeu de données."""
    dimension = 128
    vector_count = 5000

    index = create_optimized_index(dimension, vector_count)
    assert isinstance(index, faiss.IndexFlatL2)
    assert index.d == dimension

def test_create_optimized_index_medium_dataset():
    """Teste la création d'un index optimisé pour un jeu de données moyen."""
    dimension = 128
    vector_count = 50000

    index = create_optimized_index(dimension, vector_count)
    assert isinstance(index, faiss.IndexIVFFlat)
    assert index.nprobe == 8

def test_create_optimized_index_large_dataset():
    """Teste la création d'un index optimisé pour un grand jeu de données."""
    dimension = 128
    vector_count = 200000

    index = create_optimized_index(dimension, vector_count)
    assert hasattr(index, 'hnsw')
    assert index.hnsw.efConstruction == 40
    assert index.hnsw.efSearch == 16

def test_configure_search_parameters_flat():
    """Teste la configuration des paramètres de recherche pour IndexFlatL2."""
    index = faiss.IndexFlatL2(128)
    faiss_service._state.index = index

    params = configure_search_parameters(k=10, precision_priority=True)
    assert params['k'] == 10

def test_configure_search_parameters_ivf():
    """Teste la configuration des paramètres de recherche pour IndexIVFFlat."""
    dimension = 128
    quantizer = faiss.IndexFlatL2(dimension)
    index = faiss.IndexIVFFlat(quantizer, dimension, 100)
    faiss_service._state.index = index

    # Test avec precision_priority=True
    params = configure_search_parameters(k=10, precision_priority=True)
    assert params['k'] == 10
    assert params['nprobe'] >= 16

    # Test avec precision_priority=False
    params = configure_search_parameters(k=10, precision_priority=False)
    assert params['nprobe'] == 8

@pytest.mark.asyncio
async def test_update_periodically():
    """Teste la mise à jour périodique de l'index."""
    mock_load = MagicMock(return_value=(FakeFaissIndex(384), {"0": {"content": "test"}}))
    mock_sleep = AsyncMock()
    stop_event = asyncio.Event()

    with (
        patch('app.services.faiss_service.load_index', mock_load),
        patch('asyncio.sleep', mock_sleep)
    ):
        # Démarrer la tâche de mise à jour
        task = asyncio.create_task(faiss_service.update_periodically(stop_event))

        # Attendre un peu pour laisser la tâche s'exécuter
        await asyncio.sleep(0.1)

        # Arrêter la tâche
        stop_event.set()
        await task

        # Vérifications
        mock_load.assert_called_once()
        mock_sleep.assert_called_once()

@pytest.mark.asyncio
async def test_update_periodically_error():
    """Teste la gestion des erreurs dans la mise à jour périodique."""
    mock_load = MagicMock(side_effect=FAISSLoadError("Test error"))
    mock_sleep = AsyncMock()
    stop_event = asyncio.Event()

    with (
        patch('app.services.faiss_service.load_index', mock_load),
        patch('asyncio.sleep', mock_sleep)
    ):
        # Démarrer la tâche de mise à jour
        task = asyncio.create_task(faiss_service.update_periodically(stop_event))

        # Attendre un peu pour laisser la tâche s'exécuter
        await asyncio.sleep(0.1)

        # Arrêter la tâche
        stop_event.set()
        await task

        # Vérifications
        mock_load.assert_called_once()
        mock_sleep.assert_called_once()

def test_download_from_s3_error():
    """Teste la gestion des erreurs lors du téléchargement depuis S3."""
    mock_s3 = MagicMock()
    mock_s3.download_file.side_effect = ClientError(
        {"Error": {"Code": "NoSuchBucket", "Message": "The bucket does not exist"}},
        "GetObject"
    )

    with (
        patch('boto3.client', MagicMock(return_value=mock_s3)),
        pytest.raises(FAISSSyncError)
    ):
        faiss_service._download_from_s3("test.faiss", "test.json")

def test_load_document_store_invalid_json():
    """Teste le chargement d'un document store avec un JSON invalide."""
    with (
        tempfile.NamedTemporaryFile(mode='w') as temp_file,
        pytest.raises(FAISSLoadError)
    ):
        temp_file.write("Invalid JSON")
        temp_file.flush()
        faiss_service._load_document_store(temp_file.name)

def test_retrieve_similar_documents_with_cache():
    """Teste la recherche de documents similaires avec cache."""
    mock_cache = MagicMock()
    mock_cache.get_search_results.return_value = [{"content": "cached result"}]

    with patch('app.services.faiss_service.get_cache_instance', return_value=mock_cache):
        results = faiss_service.retrieve_similar_documents("test query", use_cache=True)
        assert len(results) == 1
        assert results[0]["content"] == "cached result"
        mock_cache.get_search_results.assert_called_once_with("test query")

def test_retrieve_similar_documents_dimension_mismatch():
    """Teste la gestion des différences de dimensions dans la recherche."""
    # Créer un index avec une dimension différente
    index = faiss.IndexFlatL2(256)
    faiss_service._state.index = index
    faiss_service._state.document_store = {"0": {"content": "test"}}

    with patch('app.services.embedding_service.generate_query_vector', return_value=np.zeros((1, 128))):
        results = faiss_service.retrieve_similar_documents("test query")
        assert isinstance(results, list)

def test_save_faiss_index_local_environment(tmp_path):
    """Teste la sauvegarde de l'index en environnement local."""
    dimension = 128
    index = faiss.IndexFlatL2(dimension)
    mapping = {"1": {"content": "test"}}

    with patch('app.services.faiss_service.is_local_environment', return_value=True):
        faiss_service.save_faiss_index(index, mapping, str(tmp_path))
        assert os.path.exists(os.path.join(str(tmp_path), settings.FAISS_INDEX_FILE))
        assert os.path.exists(os.path.join(str(tmp_path), settings.FAISS_METADATA_FILE))

def test_get_local_path():
    """Teste la génération des chemins locaux."""
    with patch('app.services.faiss_service.is_local_environment', return_value=True):
        path = faiss_service._get_local_path("test.file")
        assert isinstance(path, str)
        assert "test.file" in path
