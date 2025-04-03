"""Tests unitaires pour le module git_utils."""

import os
from unittest.mock import patch, Mock

import pytest
from github import Github, Repository, ContentFile
from github.GithubException import GithubException

from app.utils.git_utils import (
    get_github_client,
    get_repository,
    read_file_content,
    read_repository_content,
)


@pytest.fixture
def mock_github_client():
    """Fixture pour mocker le client GitHub."""
    with patch("github.Github") as mock_github:
        mock_instance = Mock(spec=Github)
        mock_github.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_repository():
    """Fixture pour mocker un repository GitHub."""
    mock_repo = Mock(spec=Repository.Repository)
    return mock_repo


def test_get_github_client_with_pat():
    """Teste la création du client GitHub avec un PAT."""
    with patch.dict(os.environ, {"GITHUB_PAT": "test_token"}):
        client = get_github_client()
        assert isinstance(client, Github)


def test_get_github_client_with_actions():
    """Teste la création du client GitHub avec GitHub Actions."""
    with patch.dict(os.environ, {"GITHUB_ACTIONS": "true", "GITHUB_TOKEN": "actions_token"}):
        client = get_github_client()
        assert isinstance(client, Github)


def test_get_repository_success(mock_github_client):
    """Teste la récupération réussie d'un repository."""
    mock_repo = Mock(spec=Repository.Repository)
    mock_github_client.get_repo.return_value = mock_repo

    result = get_repository("owner", "repo")
    assert result == mock_repo
    mock_github_client.get_repo.assert_called_once_with("owner/repo")


def test_get_repository_error(mock_github_client):
    """Teste la gestion d'erreur lors de la récupération d'un repository."""
    mock_github_client.get_repo.side_effect = GithubException(404, "Not Found")

    result = get_repository("owner", "nonexistent")
    assert result is None


def test_read_file_content_success(mock_repository):
    """Teste la lecture réussie du contenu d'un fichier."""
    mock_content = Mock(spec=ContentFile.ContentFile)
    mock_content.content = "SGVsbG8gV29ybGQ="  # "Hello World" en base64
    mock_repository.get_contents.return_value = mock_content

    content = read_file_content(mock_repository, "test.txt")
    assert content == "Hello World"
    mock_repository.get_contents.assert_called_once_with("test.txt")


def test_read_file_content_directory(mock_repository):
    """Teste la gestion d'un chemin qui pointe vers un dossier."""
    mock_repository.get_contents.return_value = []

    content = read_file_content(mock_repository, "dir/")
    assert content is None


def test_read_repository_content_success(mock_github_client, mock_repository):
    """Teste la lecture réussie du contenu d'un repository."""
    mock_file = Mock(spec=ContentFile.ContentFile)
    mock_file.path = "test.txt"
    mock_file.type = "file"
    mock_content = Mock(spec=ContentFile.ContentFile)
    mock_content.content = "SGVsbG8gV29ybGQ="  # "Hello World" en base64

    mock_repository.get_contents.side_effect = [
        [mock_file],  # Premier appel pour lister les fichiers
        mock_content  # Deuxième appel pour lire le contenu
    ]
    mock_github_client.get_repo.return_value = mock_repository

    result = read_repository_content("owner/repo")
    assert len(result) == 1
    assert result[0] == ("test.txt", "Hello World")


def test_read_repository_content_invalid_format():
    """Teste la gestion d'un format de repository invalide."""
    result = read_repository_content("invalid-format")
    assert result == []


def test_read_repository_content_error(mock_github_client):
    """Teste la gestion d'erreur lors de la lecture du repository."""
    mock_github_client.get_repo.side_effect = GithubException(404, "Not Found")

    result = read_repository_content("owner/nonexistent")
    assert result == []