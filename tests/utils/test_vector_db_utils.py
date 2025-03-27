"""Tests pour les utilitaires de base de données vectorielle."""

import pytest

from app.utils.vector_db_utils import process_documents_for_chroma, process_documents_for_faiss, segment_text

def test_process_documents_for_chroma_empty() -> None:
    """Test avec une liste de documents vide."""
    assert process_documents_for_chroma([]) == []

def test_process_documents_for_chroma_single_segment() -> None:
    """Test avec un document simple."""
    text = "Short document"  # Assez court pour tenir dans un seul segment
    documents = [("test.md", text)]
    processed = process_documents_for_chroma(documents)

    assert len(processed) == 1
    assert isinstance(processed[0], dict)
    assert "id" in processed[0]
    assert "text" in processed[0]
    assert "metadata" in processed[0]

def test_process_documents_for_chroma_multiple_segments() -> None:
    """Test avec un document qui nécessite plusieurs segments."""
    # Créer deux sections distinctes
    text = "## Section 1\n" + ("A" * 450) + "\n\n## Section 2\n" + ("B" * 450)
    documents = [("test.md", text)]
    processed = process_documents_for_chroma(documents)

    assert len(processed) >= 2  # Au moins 2 segments à cause des sections
    assert all(isinstance(doc, dict) for doc in processed)
    assert all("id" in doc for doc in processed)
    assert all("text" in doc for doc in processed)
    assert all("metadata" in doc for doc in processed)
    # Maintenant on vérifie que chaque section est segmentée correctement
    assert any("Section 1" in doc["text"] for doc in processed)
    assert any("Section 2" in doc["text"] for doc in processed)
    assert any("A" * 50 in doc["text"] for doc in processed)  # Vérifie qu'une partie du contenu est présente
    assert any("B" * 50 in doc["text"] for doc in processed)  # Vérifie qu'une partie du contenu est présente

def test_process_documents_for_chroma_metadata() -> None:
    """Test que les métadonnées sont correctement incluses."""
    documents = [("docs/test.md", "Test content")]
    processed = process_documents_for_chroma(documents)

    assert len(processed) == 1
    assert processed[0]["metadata"]["source"] == "docs/test.md"
    assert "segment_id" in processed[0]["metadata"]

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