"""Tests unitaires pour le module document_utils.

Ce module teste les fonctionnalités de lecture et de filtrage des fichiers.
"""

import os
import tempfile
import pytest
from unittest.mock import patch, mock_open
from typing import Generator, Any

from app.utils.document_utils import (
    is_file_relevant,
    read_code_file,
    read_relevant_files,
    EXCLUDED_EXTENSIONS,
    INCLUDED_CODE_EXTENSIONS,
)

@pytest.fixture
def temp_directory() -> Generator[str, Any, None]:
    """Crée un répertoire temporaire pour les tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def sample_files(temp_directory: str) -> list[tuple[str, str]]:
    """Crée des fichiers de test dans le répertoire temporaire."""
    files = [
        ("test.py", "print('Hello')"),
        ("test.md", "# Test"),
        ("test.txt", "Text content"),
        ("test.jpg", b"Binary content"),
        ("test.pyc", b"Compiled Python"),
    ]

    for filename, content in files:
        file_path = os.path.join(temp_directory, filename)
        mode = "w" if isinstance(content, str) else "wb"
        with open(file_path, mode) as f:
            f.write(content)

    return [(os.path.join(temp_directory, f[0]), f[1]) for f in files]

def test_is_file_relevant_included_extensions() -> None:
    """Teste la détection des fichiers à inclure."""
    for ext in INCLUDED_CODE_EXTENSIONS:
        test_file = f"test{ext}"
        assert is_file_relevant(test_file), f"Le fichier {test_file} devrait être inclus"

def test_is_file_relevant_excluded_extensions() -> None:
    """Teste la détection des fichiers à exclure."""
    for ext in EXCLUDED_EXTENSIONS:
        test_file = f"test{ext}"
        assert not is_file_relevant(test_file), f"Le fichier {test_file} devrait être exclu"

def test_is_file_relevant_unknown_extension() -> None:
    """Teste la détection des fichiers avec une extension inconnue."""
    with tempfile.NamedTemporaryFile(suffix=".unknown", mode="w") as temp_file:
        temp_file.write("Test content")
        temp_file.flush()
        assert is_file_relevant(temp_file.name), "Le fichier texte devrait être inclus"

def test_read_code_file_utf8() -> None:
    """Teste la lecture d'un fichier avec encodage UTF-8."""
    content = "print('Hello, 世界')"
    mock = mock_open(read_data=content)

    with patch("builtins.open", mock):
        result = read_code_file("test.py")
        assert result == content
        mock.assert_called_once_with("test.py", encoding="utf-8")

def test_read_code_file_latin1_fallback() -> None:
    """Teste le fallback vers l'encodage Latin-1."""
    content = "print('Hello')"
    mock = mock_open(read_data=content)

    # Simuler une erreur UTF-8 puis un succès avec Latin-1
    mock.side_effect = [UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid'), mock.return_value]

    with patch("builtins.open", mock):
        result = read_code_file("test.py")
        assert result == content
        assert mock.call_args_list[1][1]["encoding"] == "latin-1"

def test_read_code_file_error_handling() -> None:
    """Teste la gestion des erreurs de lecture."""
    with patch("builtins.open", side_effect=Exception("Test error")):
        result = read_code_file("nonexistent.py")
        assert result == ""

def test_read_relevant_files(temp_directory: str, sample_files: list[tuple[str, str]]) -> None:
    """Teste la lecture des fichiers pertinents d'un répertoire."""
    relevant_files = read_relevant_files(temp_directory)

    # Vérifier que seuls les fichiers pertinents sont inclus
    assert len(relevant_files) == 3  # .py, .md, .txt

    # Vérifier que les fichiers binaires sont exclus
    binary_files = [f for f, _ in relevant_files if f.endswith(('.jpg', '.pyc'))]
    assert not binary_files, "Les fichiers binaires ne devraient pas être inclus"

    # Vérifier le contenu des fichiers
    for file_path, content in relevant_files:
        if file_path.endswith('.py'):
            assert "print('Hello')" in content
        elif file_path.endswith('.md'):
            assert "# Test" in content
        elif file_path.endswith('.txt'):
            assert "Text content" in content

def test_read_relevant_files_empty_directory(temp_directory: str) -> None:
    """Teste la lecture d'un répertoire vide."""
    files = read_relevant_files(temp_directory)
    assert not files, "Un répertoire vide devrait retourner une liste vide"

def test_read_relevant_files_with_subdirectories(temp_directory: str) -> None:
    """Teste la lecture récursive des sous-répertoires."""
    # Créer une structure de répertoires
    subdir = os.path.join(temp_directory, "subdir")
    os.makedirs(subdir)

    # Créer des fichiers dans le sous-répertoire
    with open(os.path.join(subdir, "test.py"), "w") as f:
        f.write("print('SubdirTest')")

    files = read_relevant_files(temp_directory)
    assert any("subdir" in f[0] for f in files), "Les fichiers des sous-répertoires devraient être inclus"

def test_read_relevant_files_with_empty_files(temp_directory: str) -> None:
    """Teste le comportement avec des fichiers vides."""
    # Créer un fichier vide
    empty_file = os.path.join(temp_directory, "empty.py")
    with open(empty_file, "w") as f:
        f.write("")

    files = read_relevant_files(temp_directory)
    assert not any(f[0] == empty_file for f in files), "Les fichiers vides devraient être ignorés"

def test_read_relevant_files_error_logging(temp_directory: str) -> None:
    """Teste la journalisation des erreurs lors de la lecture des fichiers."""
    with (
        patch("app.utils.document_utils.read_code_file", side_effect=Exception("Test error")),
        patch("app.utils.document_utils.logger") as mock_logger
    ):
        # Créer un fichier de test
        test_file = os.path.join(temp_directory, "test.py")
        with open(test_file, "w") as f:
            f.write("test")

        read_relevant_files(temp_directory)

        # Vérifier que l'erreur a été journalisée
        mock_logger.warning.assert_called_once()
        assert "Erreur lors de la lecture du fichier" in mock_logger.warning.call_args[0][0]