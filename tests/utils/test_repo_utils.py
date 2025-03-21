"""
Tests unitaires pour le module repo_utils.
"""

import os
import subprocess
import sys
import tempfile
from unittest.mock import patch, MagicMock, mock_open

import pytest

from app.utils.repo_utils import clone_or_update_repo


@pytest.fixture
def mock_repo_url():
    """Fixture pour l'URL du dépôt de test."""
    return "https://github.com/user/repo.git"


@pytest.fixture
def mock_repo_dir():
    """Fixture pour le répertoire de test."""
    return "/test/repo"


def test_clone_new_repository(mock_repo_url, mock_repo_dir):
    """
    Teste le clonage d'un nouveau dépôt quand le répertoire cible n'existe pas.
    """
    with patch("subprocess.run") as mock_run, \
         patch("os.path.exists") as mock_exists, \
         patch("os.makedirs") as mock_makedirs:
        
        # Configure les mocks pour simuler un répertoire qui n'existe pas
        mock_exists.return_value = False
        mock_run.return_value.returncode = 0
        
        # Exécute la fonction
        result = clone_or_update_repo(mock_repo_url, mock_repo_dir)
        
        # Vérifie que le bon chemin est retourné
        assert result == os.path.abspath(mock_repo_dir)
        
        # Vérifie que git clone a été appelé avec les bons arguments
        mock_run.assert_called_once_with(
            ["git", "clone", mock_repo_url, os.path.abspath(mock_repo_dir)],
            check=True
        )


def test_update_existing_repository(mock_repo_url, mock_repo_dir):
    """
    Teste la mise à jour d'un dépôt existant.
    """
    mock_write_test = mock_open()
    test_file_path = os.path.join(os.path.abspath(mock_repo_dir), ".write_test")
    
    with patch("os.path.exists") as mock_exists, \
         patch("subprocess.run") as mock_run, \
         patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_write_test) as mock_file, \
         patch("os.remove") as mock_remove:
        
        # Configure les mocks pour simuler un répertoire existant avec accès en écriture
        def mock_exists_side_effect(path):
            if path.endswith(".git"):
                return True
            if path == os.path.abspath(mock_repo_dir):
                return True
            if path == os.path.dirname(os.path.abspath(mock_repo_dir)):
                return True
            return False

        mock_exists.side_effect = mock_exists_side_effect
        mock_run.return_value.returncode = 0
        
        # Exécute la fonction
        result = clone_or_update_repo(mock_repo_url, mock_repo_dir)
        
        # Vérifie que le test d'écriture a été effectué
        mock_file.assert_called_once_with(test_file_path, "w")
        mock_file().write.assert_called_once_with("test")
        mock_remove.assert_called_once_with(test_file_path)
        
        # Vérifie que le bon chemin est retourné
        assert result == os.path.abspath(mock_repo_dir)
        
        # Vérifie que git pull a été appelé avec les bons arguments
        mock_run.assert_called_once_with(
            ["git", "-C", os.path.abspath(mock_repo_dir), "pull"],
            check=True
        )


def test_clone_error_handling(mock_repo_url, mock_repo_dir):
    """
    Teste la gestion des erreurs lors du clonage.
    """
    with patch("os.path.exists") as mock_exists, \
         patch("subprocess.run") as mock_run, \
         patch("os.makedirs"), \
         pytest.raises(SystemExit) as exit_info:
        
        # Configure les mocks pour simuler une erreur de clonage
        mock_exists.return_value = False
        mock_run.side_effect = subprocess.CalledProcessError(1, "git clone")
        
        # Exécute la fonction
        clone_or_update_repo(mock_repo_url, mock_repo_dir)
        
        # Vérifie que le programme se termine avec le code 1
        assert exit_info.value.code == 1


def test_update_error_handling(mock_repo_url, mock_repo_dir):
    """
    Teste la gestion des erreurs lors de la mise à jour.
    """
    with patch("os.path.exists") as mock_exists, \
         patch("subprocess.run") as mock_run, \
         patch("os.makedirs"), \
         patch("builtins.open", mock_open()) as mock_file, \
         pytest.raises(SystemExit) as exit_info:
        
        # Configure les mocks pour simuler une erreur de mise à jour
        def mock_exists_side_effect(path):
            if path == mock_repo_dir or path == os.path.join(os.path.abspath(mock_repo_dir), ".git"):
                return True
            return False

        mock_exists.side_effect = mock_exists_side_effect
        mock_run.side_effect = subprocess.CalledProcessError(1, "git pull")
        
        # Exécute la fonction
        clone_or_update_repo(mock_repo_url, mock_repo_dir)
        
        # Vérifie que le programme se termine avec le code 1
        assert exit_info.value.code == 1


def test_fallback_to_temp_directory(mock_repo_url, mock_repo_dir):
    """
    Teste le fallback vers un répertoire temporaire quand le répertoire cible
    n'est pas accessible en écriture.
    """
    with patch("os.path.exists") as mock_exists, \
         patch("os.makedirs") as mock_makedirs, \
         patch("tempfile.gettempdir") as mock_tempdir, \
         patch("subprocess.run") as mock_run, \
         patch("os.getpid") as mock_getpid, \
         patch("builtins.open") as mock_open:
        
        # Configure les mocks
        mock_exists.return_value = True
        mock_open.side_effect = IOError()
        mock_tempdir.return_value = "/tmp"
        mock_getpid.return_value = 12345
        mock_run.return_value.returncode = 0
        
        # Exécute la fonction
        result = clone_or_update_repo(mock_repo_url, mock_repo_dir)
        
        # Vérifie que le chemin retourné est dans le répertoire temporaire
        expected_temp_dir = os.path.join("/tmp", "repo_clone_12345")
        assert result == expected_temp_dir
        
        # Vérifie que le répertoire temporaire a été créé
        mock_makedirs.assert_called_with(expected_temp_dir, exist_ok=True)


@pytest.mark.parametrize("error_type", [IOError, OSError])
def test_directory_creation_errors(mock_repo_url, mock_repo_dir, error_type):
    """
    Teste la gestion des erreurs lors de la création des répertoires.
    """
    with patch("os.path.exists") as mock_exists, \
         patch("os.makedirs") as mock_makedirs, \
         patch("tempfile.gettempdir") as mock_tempdir, \
         patch("subprocess.run") as mock_run, \
         patch("os.getpid") as mock_getpid:
        
        # Configure les mocks pour simuler une erreur de création puis un succès
        mock_exists.return_value = False
        mock_makedirs.side_effect = [error_type(), None]
        mock_tempdir.return_value = "/tmp"
        mock_getpid.return_value = 12345
        mock_run.return_value.returncode = 0
        
        # Exécute la fonction
        result = clone_or_update_repo(mock_repo_url, mock_repo_dir)
        
        # Vérifie que le chemin retourné est dans le répertoire temporaire
        expected_temp_dir = os.path.join("/tmp", "repo_clone_12345")
        assert result == expected_temp_dir
        
        # Vérifie que le répertoire temporaire a été créé
        assert mock_makedirs.call_args_list[-1] == ((expected_temp_dir,), {"exist_ok": True})