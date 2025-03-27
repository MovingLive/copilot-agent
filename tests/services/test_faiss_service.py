import numpy as np
import pytest
from unittest.mock import patch, MagicMock
from app.services import faiss_service
from app.core.config import settings
import os
import json
import faiss

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
