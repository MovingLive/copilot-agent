"""Tests pour les utilitaires de base de données vectorielle."""

import pytest
from pathlib import Path

from app.utils.vector_db_utils import process_documents_for_faiss, process_files_for_faiss


def test_process_documents_for_faiss() -> None:
    """Test de base pour process_documents_for_faiss."""
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
    """Test des IDs incrémentaux pour process_documents_for_faiss."""
    documents = [("test1.md", "Document 1"), ("test2.md", "Document 2")]
    processed = process_documents_for_faiss(documents)

    numeric_ids = [doc["numeric_id"] for doc in processed]
    assert numeric_ids == list(range(len(processed)))
    assert all(doc["metadata"]["file_path"] == f"test{i+1}.md"
              for i, doc in enumerate(processed))
    assert all(doc["metadata"]["segment_index"] == 0 for doc in processed)


def test_process_files_for_faiss_empty_documents() -> None:
    """Test avec des documents vides."""
    documents = [
        ("empty.md", ""),
        ("empty.py", ""),
    ]
    processed = process_files_for_faiss(documents)

    assert len(processed) == 0, "Les documents vides ne devraient pas générer de segments"


def test_process_files_for_faiss_different_file_types() -> None:
    """Test avec différents types de fichiers."""
    documents = [
        ("test.py", "def test(): return True"),
        ("test.md", "# Test\nContent"),
        ("test.txt", "Plain text"),
        ("test.json", '{"key": "value"}'),
    ]
    processed = process_files_for_faiss(documents)

    # Vérifier que chaque type de fichier est correctement traité
    file_types = {doc["metadata"]["file_type"] for doc in processed}
    expected_types = {".py", ".md", ".txt", ".json"}
    assert file_types == expected_types

    # Vérifier que les contenus sont préservés
    assert any(doc["text"] == "def test(): return True" for doc in processed)
    assert any(doc["text"] == "# Test\nContent" for doc in processed)
    assert any(doc["text"] == "Plain text" for doc in processed)
    assert any(doc["text"] == '{"key": "value"}' for doc in processed)


def test_process_files_for_faiss_custom_max_length() -> None:
    """Test avec une longueur maximale personnalisée."""
    long_text = "This is a " + "very "*100 + "long text"
    documents = [("long.txt", long_text)]
    
    # Test avec une petite longueur maximale
    processed_short = process_files_for_faiss(documents, max_length=20)
    assert len(processed_short) > 1, "Le texte devrait être divisé en plusieurs segments"
    assert all(len(doc["text"]) <= 20 for doc in processed_short)

    # Test avec une grande longueur maximale
    processed_long = process_files_for_faiss(documents, max_length=1000)
    assert len(processed_long) <= len(processed_short), "Moins de segments avec une plus grande longueur maximale"


def test_process_files_for_faiss_segment_metadata() -> None:
    """Test des métadonnées générées pour différents types de fichiers."""
    documents = [
        ("src/test.py", "print('test')"),
        ("docs/test.md", "# Documentation"),
    ]
    processed = process_files_for_faiss(documents)

    for doc in processed:
        assert "original_id" in doc["metadata"]
        assert "file_path" in doc["metadata"]
        assert "file_type" in doc["metadata"]
        assert "segment_index" in doc["metadata"]
        assert "segment_id" in doc["metadata"]
        assert "content" in doc["metadata"]
        
        # Vérifier que original_id est correctement formé
        file_stem = Path(doc["metadata"]["file_path"]).stem
        segment_idx = doc["metadata"]["segment_index"]
        assert doc["metadata"]["original_id"] == f"{file_stem}_{segment_idx}"


def test_process_files_for_faiss_with_special_characters() -> None:
    """Test avec des caractères spéciaux dans le contenu et les noms de fichiers."""
    documents = [
        ("test-é.py", "print('éèà')"),
        ("test_中文.txt", "Unicode 测试"),
    ]
    processed = process_files_for_faiss(documents)

    assert len(processed) == 2
    assert any("éèà" in doc["text"] for doc in processed)
    assert any("测试" in doc["text"] for doc in processed)


def test_process_files_for_faiss_long_paths() -> None:
    """Test avec des chemins de fichiers longs et complexes."""
    long_path = "very/long/path/with/multiple/subdirectories/test.py"
    documents = [(long_path, "test content")]
    processed = process_files_for_faiss(documents)

    assert len(processed) == 1
    assert processed[0]["metadata"]["file_path"] == long_path
    assert Path(processed[0]["metadata"]["file_path"]).suffix == ".py"