"""Tests pour les utilitaires de traitement Markdown."""

from pathlib import Path
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
    """Test de la segmentation d'un texte court."""
    # Un texte court devrait produire un seul segment
    text = "This is a short text."
    segments = segment_text(text, max_length=500)
    assert len(segments) == 1
    # La section Introduction est maintenant ajoutée automatiquement
    assert "This is a short text." in segments[0]

def test_segment_text_long_text() -> None:
    text = "## Section 1\nThis is a longer text.\n\n## Section 2\nIt has multiple paragraphs.\n\n## Section 3\nEach should be a segment."
    segments = segment_text(text)
    assert len(segments) == 3
    assert "## Section 1" in segments[0]
    assert "## Section 2" in segments[1]
    assert "## Section 3" in segments[2]

def test_segment_text_very_long_paragraph() -> None:
    """Test de segmentation d'un long texte avec titre."""
    text = "## Title\n" + "x" * 600  # Crée un texte de 600 caractères avec un titre
    segments = segment_text(text, max_length=500)
    # Avec le chevauchement intelligent, le texte est divisé en segments
    assert len(segments) >= 1
    # Vérifie que tous les segments contiennent le titre
    for segment in segments:
        assert "## Title" in segment
    # Vérifie que le contenu est présent
    content = "".join(segments)
    assert "x" * 600 in content

def test_segment_text_multiple_paragraphs() -> None:
    """Test de segmentation avec plusieurs paragraphes."""
    text = "\n".join([
        "## Section 1",
        "First paragraph.",
        "",
        "Second paragraph.",
        "",
        "## Section 2",
        "Third paragraph."
    ])
    segments = segment_text(text)
    assert len(segments) >= 2  # Au moins 2 segments pour 2 sections
    # Vérifie que les titres de section sont préservés
    assert any("## Section 1" in s for s in segments)
    assert any("## Section 2" in s for s in segments)
    # Vérifie que le contenu est présent
    assert any("First paragraph" in s for s in segments)
    assert any("Third paragraph" in s for s in segments)

def test_read_markdown_files(tmp_path: Path) -> None:
    """Test de lecture des fichiers Markdown."""
    # Créer un fichier Markdown de test
    test_file = tmp_path / "test.md"
    test_content = "# Test\nThis is a test file."
    test_file.write_text(test_content)
    
    # Lire les fichiers
    documents = read_markdown_files(str(tmp_path))
    assert len(documents) == 1
    assert documents[0][0].endswith("test.md")
    assert documents[0][1] == test_content

def test_read_markdown_files_empty_dir(tmp_path: Path) -> None:
    """Test de lecture d'un répertoire vide."""
    documents = read_markdown_files(str(tmp_path))
    assert len(documents) == 0

def test_read_markdown_files_invalid_file(tmp_path: Path) -> None:
    """Test de gestion des erreurs de lecture."""
    # Créer un fichier avec des permissions en lecture-seule
    test_file = tmp_path / "test.md"
    test_file.touch()
    test_file.chmod(0o000)  # Retirer toutes les permissions
    
    try:
        documents = read_markdown_files(str(tmp_path))
        assert len(documents) == 0
    finally:
        # Restaurer les permissions pour le nettoyage
        test_file.chmod(0o666)