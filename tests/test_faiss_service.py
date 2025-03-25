import numpy as np
import pytest
from app.services import faiss_service

# ...existing code...

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

def fake_search_error(query_vector, k):
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
