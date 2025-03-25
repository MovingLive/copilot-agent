"""Tests unitaires pour le module markdown_utils."""

import pytest

from app.utils.markdown_utils import read_markdown_files, segment_text

@pytest.fixture
def sample_markdown_files(tmp_path) -> str:
    # Créer des fichiers markdown temporaires pour les tests
    file1 = tmp_path / "test1.md"
    file2 = tmp_path / "test2.md"

    file1.write_text("# Test 1\nContent 1")
    file2.write_text("# Test 2\nContent 2")

    return str(tmp_path)

def test_read_markdown_files(sample_markdown_files: str) -> None:
    documents = read_markdown_files(sample_markdown_files)
    assert len(documents) == 2
    assert all(isinstance(doc, tuple) for doc in documents)
    assert all(len(doc) == 2 for doc in documents)
    assert all(doc[1].startswith("# Test") for doc in documents)

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
    """Test de segmentation d'un long texte sans espaces."""
    text = "## Title\n" + "x" * 600  # Crée un texte de 600 caractères avec un titre
    segments = segment_text(text, max_length=500)
    assert len(segments) == 2
    assert "## Title" in segments[0]
    assert len(segments[0]) <= 500
    assert "## Title" in segments[1]  # Le titre est répété dans chaque segment

def test_segment_text_with_overlap() -> None:
    """Test de segmentation d'un long texte avec des sections titrées."""
    text = "## Section A\n" + "A" * 100 + "\n## Section B\n" + "B" * 100 + "\n## Section C\n" + "C" * 100
    segments = segment_text(text, max_length=150)

    assert len(segments) >= 2  # Au moins 2 segments dû aux titres
    assert all(len(segment) <= 150 for segment in segments)
    assert any("## Section A" in segment for segment in segments)
    assert any("## Section B" in segment for segment in segments)
    assert any("## Section C" in segment for segment in segments)