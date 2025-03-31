"""Tests pour les utilitaires de base de données vectorielle."""

import pytest

from app.utils.vector_db_utils import process_documents_for_faiss


def test_process_documents_for_faiss() -> None:
    documents = [("test.md", "## Test\n\nThis is a test document.")]
    processed = process_documents_for_faiss(documents)

    assert len(processed) == 1  # Un seul segment car le texte est court
    assert all(isinstance(doc, dict) for doc in processed)
    assert all("numeric_id" in doc for doc in processed)
    assert all("text" in doc for doc in processed)
    assert all("metadata" in doc for doc in processed)
    assert all(isinstance(doc["numeric_id"], int) for doc in processed)
    assert processed[0]["text"] == "## Test\n\nThis is a test document."
    assert processed[0]["metadata"]["file_path"] == "test.md"
    assert processed[0]["metadata"]["segment_index"] == 0

def test_process_documents_for_faiss_incremental_ids() -> None:
    documents = [("test1.md", "Document 1"), ("test2.md", "Document 2")]
    processed = process_documents_for_faiss(documents)

    numeric_ids = [doc["numeric_id"] for doc in processed]
    assert numeric_ids == list(range(len(processed)))
    assert all(doc["metadata"]["file_path"] == f"test{i+1}.md"
              for i, doc in enumerate(processed))
    assert all(doc["metadata"]["segment_index"] == 0 for doc in processed)