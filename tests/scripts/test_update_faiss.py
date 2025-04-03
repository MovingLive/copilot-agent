"""Tests pour le script update_faiss.py."""

import os
from unittest.mock import patch, Mock, MagicMock

import numpy as np
import pytest
from github import Github, Repository, ContentFile
from moto import mock_aws

from app.core.config import settings
from scripts.update_faiss import main, load_embedding_model, get_repo_list


class MockSentenceTransformer:
    """Mock pour SentenceTransformer."""
    def encode(self, texts, show_progress_bar=False):
        return np.array([[0.1, 0.2, 0.3]] * len(texts))


class MockSettings:
    """Mock pour les paramètres de configuration."""
    TEMP_FAISS_DIR = "/tmp/faiss"
    S3_BUCKET_PREFIX = "test-prefix"
    SEGMENT_MAX_LENGTH = 1000


@pytest.fixture
def mock_env_vars():
    """Fixture pour configurer les variables d'environnement."""
    with patch.dict(os.environ, {
        "REPO_URLS": '["owner/repo1", "owner/repo2"]',
        "ENV": "test",
        "S3_BUCKET_NAME": "test-bucket",
        "GITHUB_PAT": "test-token"
    }):
        yield


@pytest.fixture
def mock_github_repo():
    """Fixture pour mocker un repository GitHub."""
    mock_repo = Mock(spec=Repository.Repository)
    mock_file = Mock(spec=ContentFile.ContentFile)
    mock_file.path = "test.md"
    mock_file.type = "file"
    mock_file.content = "SGVsbG8gV29ybGQ="  # "Hello World" en base64
    mock_repo.get_contents.return_value = [mock_file]
    return mock_repo


@pytest.fixture
def mock_github_client(mock_github_repo):
    """Fixture pour mocker le client GitHub."""
    with patch("github.Github") as mock_github:
        mock_instance = Mock(spec=Github)
        mock_instance.get_repo.return_value = mock_github_repo
        mock_github.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_embeddings():
    """Fixture pour les embeddings."""
    return np.array([[0.1, 0.2, 0.3]])


def test_get_repo_list():
    """Teste la fonction get_repo_list."""
    with patch.dict(os.environ, {"REPO_URLS": '["owner/repo1", "owner/repo2"]'}):
        repos = get_repo_list()
        assert repos == ["owner/repo1", "owner/repo2"]

    with patch.dict(os.environ, {"REPO_URLS": "owner/repo1,owner/repo2"}):
        repos = get_repo_list()
        assert repos == ["owner/repo1", "owner/repo2"]

    with patch.dict(os.environ, {"REPO_URLS": "[]"}):
        repos = get_repo_list()
        assert repos == []


def test_main_workflow(mock_github_client, mock_env_vars, mock_embeddings):
    """Teste le workflow principal du script."""
    with (
        patch("scripts.update_faiss.load_embedding_model") as mock_model,
        patch("scripts.update_faiss.process_files_for_faiss") as mock_process,
        patch("scripts.update_faiss.save_faiss_index") as mock_save,
        patch("scripts.update_faiss.export_data") as mock_export,
    ):
        # Configuration des mocks
        mock_model.return_value = MockSentenceTransformer()
        mock_process.return_value = [{"numeric_id": 1, "text": "test", "metadata": {}}]

        # Exécution de la fonction principale
        main()

        # Vérifications
        mock_github_client.get_repo.assert_called()
        mock_process.assert_called_once()
        mock_save.assert_called_once_with(MagicMock(), MagicMock(), settings.TEMP_FAISS_DIR)
        mock_export.assert_called_once()


@mock_aws
def test_s3_export(mock_github_client, mock_env_vars, mock_embeddings):
    """Teste l'exportation des données vers S3."""
    import boto3

    # Configurer le mock S3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")

    with (
        patch("scripts.update_faiss.process_files_for_faiss") as mock_process,
        patch("scripts.update_faiss.save_faiss_index") as mock_save,
        patch("scripts.update_faiss.export_data") as mock_export,
    ):
        # Configuration des mocks
        mock_process.return_value = [{"numeric_id": 1, "text": "test", "metadata": {}}]

        # Exécution de la fonction principale
        main()

        # Vérifications
        mock_save.assert_called_once_with(MagicMock(), MagicMock(), settings.TEMP_FAISS_DIR)
        mock_export.assert_called_once_with(settings.TEMP_FAISS_DIR, settings.S3_BUCKET_PREFIX)


def test_error_handling(mock_env_vars):
    """Teste la gestion des erreurs."""
    with (
        patch("scripts.update_faiss.read_repository_content", return_value=[]),
        patch("scripts.update_faiss.logging.error") as mock_error,
        pytest.raises(ValueError, match="Aucun document n'a été lu depuis les repositories"),
    ):
        main()
        mock_error.assert_called()
