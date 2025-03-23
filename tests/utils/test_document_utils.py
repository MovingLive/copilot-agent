"""
Tests unitaires pour le module document_utils.
"""

import os
import tempfile
from unittest.mock import patch

import pytest

from app.utils.document_utils import (
    clone_or_update_repo,
    process_documents_for_chroma,
    process_documents_for_faiss,
    read_markdown_files,
    segment_text,
)


# Fixtures
@pytest.fixture
def sample_repo_url() -> str:
    return "https://github.com/test/repo.git"


@pytest.fixture
def sample_repo_dir() -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_markdown_content() -> str:
    return """# Title

This is a test paragraph.

Another paragraph with some content.
"""


@pytest.fixture
def sample_markdown_files(tmp_path) -> str:
    # Créer des fichiers markdown temporaires pour les tests
    file1 = tmp_path / "test1.md"
    file2 = tmp_path / "test2.md"

    file1.write_text("# Test 1\nContent 1")
    file2.write_text("# Test 2\nContent 2")

    return str(tmp_path)


@pytest.fixture
def disable_skip_git_calls():
    """Désactiver temporairement la variable SKIP_GIT_CALLS pour les tests qui vérifient les appels git."""
    old_value = os.environ.get("SKIP_GIT_CALLS")
    os.environ["SKIP_GIT_CALLS"] = "false"
    yield
    if old_value is not None:
        os.environ["SKIP_GIT_CALLS"] = old_value
    else:
        del os.environ["SKIP_GIT_CALLS"]


# Tests pour clone_or_update_repo
def test_clone_repo_new_directory(sample_repo_url: str, sample_repo_dir: str, disable_skip_git_calls) -> None:
    with patch("subprocess.run") as mock_run:
        result = clone_or_update_repo(sample_repo_url, sample_repo_dir)
        assert os.path.exists(result)
        mock_run.assert_called_once_with(
            ["git", "clone", sample_repo_url, sample_repo_dir], check=True
        )


def test_update_existing_repo(sample_repo_url: str, sample_repo_dir: str, disable_skip_git_calls) -> None:
    # Simuler un repo existant
    os.makedirs(os.path.join(sample_repo_dir, ".git"))

    with patch("subprocess.run") as mock_run:
        result = clone_or_update_repo(sample_repo_url, sample_repo_dir)
        assert result == sample_repo_dir
        mock_run.assert_called_once_with(
            ["git", "-C", sample_repo_dir, "pull"], check=True
        )


# Tests pour read_markdown_files
def test_read_markdown_files(sample_markdown_files: str) -> None:
    documents = read_markdown_files(sample_markdown_files)
    assert len(documents) == 2
    assert all(isinstance(doc, tuple) for doc in documents)
    assert all(len(doc) == 2 for doc in documents)
    assert all(doc[1].startswith("# Test") for doc in documents)


# Tests pour segment_text
def test_segment_text_short_text() -> None:
    text = "This is a short text."
    segments = segment_text(text, max_length=100)
    assert len(segments) == 1
    assert segments[0] == text


def test_segment_text_long_text() -> None:
    text = "## Section 1\nThis is a longer text.\n\n## Section 2\nIt has multiple paragraphs.\n\n## Section 3\nEach should be a segment."
    segments = segment_text(text)
    assert len(segments) == 3
    assert "## Section 1" in segments[0]
    assert "## Section 2" in segments[1]
    assert "## Section 3" in segments[2]


def test_segment_text_very_long_paragraph() -> None:
    """
    Test de segmentation d'un long texte sans espaces.
    Le texte de test fait 600 caractères et devrait être découpé en segments
    selon la longueur maximale spécifiée.
    """
    text = "## Title\n" + "x" * 600  # Crée un texte de 600 caractères avec un titre
    segments = segment_text(text, max_length=500, overlap=0)
    assert len(segments) == 2
    assert "## Title" in segments[0]
    assert len(segments[0]) <= 500
    assert "## Title" in segments[1]  # Le titre est répété dans chaque segment


def test_segment_text_with_overlap() -> None:
    """
    Test de segmentation d'un long texte avec des sections titrées.
    """
    text = "## Section A\n" + "A" * 100 + "\n## Section B\n" + "B" * 100 + "\n## Section C\n" + "C" * 100
    segments = segment_text(text, max_length=150, overlap=50)
    
    assert len(segments) >= 2  # Au moins 2 segments dû aux titres
    
    # Vérifier que chaque segment ne dépasse pas la longueur maximale
    assert all(len(segment) <= 150 for segment in segments)
    
    # Vérifier que les segments contiennent les titres de section
    assert any("## Section A" in segment for segment in segments)
    assert any("## Section B" in segment for segment in segments)
    assert any("## Section C" in segment for segment in segments)


# Tests pour process_documents_for_chroma
def test_process_documents_for_chroma() -> None:
    documents = [("test.md", "## Test\n\nThis is a test document.")]  # Utilisation de ## pour le titre
    processed = process_documents_for_chroma(documents)

    assert len(processed) == 1  # Un seul segment car le texte est court
    assert all(isinstance(doc, dict) for doc in processed)
    assert all("id" in doc for doc in processed)
    assert all("text" in doc for doc in processed)
    assert all("metadata" in doc for doc in processed)


# Tests pour process_documents_for_faiss
def test_process_documents_for_faiss() -> None:
    documents = [("test.md", "## Test\n\nThis is a test document.")]  # Utilisation de ## pour le titre
    processed = process_documents_for_faiss(documents)

    assert len(processed) == 1  # Un seul segment car le texte est court
    assert all(isinstance(doc, dict) for doc in processed)
    assert all("numeric_id" in doc for doc in processed)
    assert all("text" in doc for doc in processed)
    assert all("metadata" in doc for doc in processed)
    assert all(isinstance(doc["numeric_id"], int) for doc in processed)


def test_process_documents_for_faiss_incremental_ids() -> None:
    documents = [("test1.md", "Document 1"), ("test2.md", "Document 2")]
    processed = process_documents_for_faiss(documents)

    numeric_ids = [doc["numeric_id"] for doc in processed]
    assert numeric_ids == list(range(len(processed)))
