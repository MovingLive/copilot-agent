"""
Tests unitaires pour le script update_chroma.py
"""

import os
from unittest.mock import MagicMock, patch

import pytest
from chromadb.config import Settings
from moto import mock_aws

from scripts.update_chroma import (
    CHROMA_PERSIST_DIR,
    COLLECTION_NAME,
    REPO_DIR,
    REPO_URL,
    SEGMENT_MAX_LENGTH,
    main,
)


@pytest.fixture
def mock_documents():
    """Fixture pour simuler les documents de test."""
    return [
        {
            "id": "doc1",
            "content": "Test document 1",
            "path": "docs/test1.md",
        },
        {
            "id": "doc2",
            "content": "Test document 2",
            "path": "docs/test2.md",
        },
    ]


@pytest.fixture
def mock_processed_docs():
    """Fixture pour simuler les documents traités."""
    return [
        {
            "id": "segment_1",
            "text": "Test document 1",
            "metadata": {"source": "docs/test1.md", "segment": 0},
        },
        {
            "id": "segment_2",
            "text": "Test document 2",
            "metadata": {"source": "docs/test2.md", "segment": 0},
        },
    ]


@pytest.fixture
def mock_chromadb_client():
    """Fixture pour simuler un client ChromaDB."""
    client = MagicMock()
    collection = MagicMock()
    client.create_collection.return_value = collection
    return client


@pytest.fixture
def mock_env_vars():
    """Fixture pour configurer les variables d'environnement de test."""
    with patch.dict(
        os.environ,
        {
            "ENV": "test",
            "REPO_URL": "https://github.com/test/repo.git",
            "REPO_DIR": "test_repo",
            "S3_BUCKET_NAME": "test-bucket",
        },
    ):
        yield


def test_main_workflow(
    mock_documents, mock_processed_docs, mock_chromadb_client, mock_env_vars
):
    """
    Teste le flux principal de la fonction main.
    """
    with (
        patch("scripts.update_chroma.clone_or_update_repo") as mock_clone,
        patch("scripts.update_chroma.read_markdown_files") as mock_read,
        patch("scripts.update_chroma.process_documents_for_chroma") as mock_process,
        patch("scripts.update_chroma.chromadb.Client") as mock_client_class,
        patch("scripts.update_chroma.export_data") as mock_export,
        patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
        ) as mock_ef,
    ):
        # Configuration des mocks
        mock_clone.return_value = "test_repo_path"
        mock_read.return_value = mock_documents
        mock_process.return_value = mock_processed_docs
        mock_client_class.return_value = mock_chromadb_client

        # Exécution de la fonction principale
        main()

        # Vérifications
        mock_clone.assert_called_once_with(REPO_URL, REPO_DIR)
        mock_read.assert_called_once_with("test_repo_path")
        mock_process.assert_called_once_with(mock_documents, SEGMENT_MAX_LENGTH)
        mock_client_class.assert_called_once()
        mock_chromadb_client.create_collection.assert_called_once()
        mock_export.assert_called_once()


def test_chroma_collection_creation(mock_chromadb_client, mock_env_vars):
    """
    Teste la création et la configuration de la collection ChromaDB.
    """
    with (
        patch("scripts.update_chroma.chromadb.Client") as mock_client_class,
        patch(
            "chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction"
        ) as mock_ef,
    ):
        mock_client_class.return_value = mock_chromadb_client

        # Tente de supprimer la collection existante
        mock_chromadb_client.delete_collection.side_effect = ValueError()

        # Exécution de la fonction principale
        main()

        # Vérifications
        mock_client_class.assert_called_once_with(
            Settings(persist_directory=CHROMA_PERSIST_DIR, anonymized_telemetry=False)
        )
        mock_chromadb_client.create_collection.assert_called_once_with(
            name=COLLECTION_NAME,
            embedding_function=mock_ef.return_value,
        )


@mock_aws
def test_s3_export(mock_env_vars):
    """
    Teste l'exportation des données vers S3.
    """
    import boto3

    # Configurer le mock S3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")

    with (
        patch("scripts.update_chroma.clone_or_update_repo"),
        patch("scripts.update_chroma.read_markdown_files"),
        patch("scripts.update_chroma.process_documents_for_chroma"),
        patch("scripts.update_chroma.chromadb.Client"),
        patch("scripts.update_chroma.export_data") as mock_export,
    ):
        # Exécution de la fonction principale
        main()

        # Vérification que l'export a été appelé
        mock_export.assert_called_once()


def test_document_processing(mock_documents, mock_env_vars):
    """
    Teste le traitement des documents.
    """
    with (
        patch("scripts.update_chroma.clone_or_update_repo"),
        patch("scripts.update_chroma.read_markdown_files") as mock_read,
        patch("scripts.update_chroma.process_documents_for_chroma") as mock_process,
        patch("scripts.update_chroma.chromadb.Client"),
        patch("scripts.update_chroma.export_data"),
    ):
        mock_read.return_value = mock_documents
        mock_process.return_value = []

        # Exécution de la fonction principale
        main()

        # Vérifications
        mock_read.assert_called_once()
        mock_process.assert_called_once_with(mock_documents, SEGMENT_MAX_LENGTH)


def test_error_handling(mock_env_vars):
    """
    Teste la gestion des erreurs.
    """
    with (
        patch("scripts.update_chroma.clone_or_update_repo") as mock_clone,
        patch("scripts.update_chroma.logging.error") as mock_error,
        pytest.raises(Exception),
    ):
        # Simuler une erreur lors du clonage
        mock_clone.side_effect = Exception("Test error")

        # Exécution de la fonction principale
        main()

        # Vérifier que l'erreur a été journalisée
        mock_error.assert_called()


def test_batch_processing(mock_chromadb_client, mock_env_vars):
    """
    Teste le traitement par lots des documents dans ChromaDB.
    """
    # Créer un grand nombre de documents pour tester le traitement par lots
    large_processed_docs = [
        {
            "id": f"segment_{i}",
            "text": f"Test document {i}",
            "metadata": {"source": f"docs/test{i}.md", "segment": 0},
        }
        for i in range(300)  # Plus que la taille du lot (200)
    ]

    with (
        patch("scripts.update_chroma.clone_or_update_repo"),
        patch("scripts.update_chroma.read_markdown_files"),
        patch("scripts.update_chroma.process_documents_for_chroma") as mock_process,
        patch("scripts.update_chroma.chromadb.Client") as mock_client_class,
        patch("scripts.update_chroma.export_data"),
    ):
        mock_process.return_value = large_processed_docs
        mock_client_class.return_value = mock_chromadb_client
        collection = mock_chromadb_client.create_collection.return_value

        # Exécution de la fonction principale
        main()

        # Vérifier que add a été appelé deux fois (300 docs / 200 par lot = 2 appels)
        assert collection.add.call_count == 2


def test_empty_documents(mock_chromadb_client, mock_env_vars):
    """
    Teste le comportement avec une liste de documents vide.
    """
    with (
        patch("scripts.update_chroma.clone_or_update_repo"),
        patch("scripts.update_chroma.read_markdown_files") as mock_read,
        patch("scripts.update_chroma.process_documents_for_chroma") as mock_process,
        patch("scripts.update_chroma.chromadb.Client") as mock_client_class,
        patch("scripts.update_chroma.export_data"),
    ):
        mock_read.return_value = []
        mock_process.return_value = []
        mock_client_class.return_value = mock_chromadb_client
        collection = mock_chromadb_client.create_collection.return_value

        # Exécution de la fonction principale
        main()

        # Vérifier que add n'a pas été appelé
        collection.add.assert_not_called()


def test_collection_deletion_error(mock_chromadb_client, mock_env_vars):
    """
    Teste la gestion d'erreur lors de la suppression de la collection.
    """
    with (
        patch("scripts.update_chroma.clone_or_update_repo"),
        patch("scripts.update_chroma.chromadb.Client") as mock_client_class,
    ):
        mock_client_class.return_value = mock_chromadb_client
        mock_chromadb_client.delete_collection.side_effect = ValueError(
            "Collection not found"
        )

        try:
            main()
        except ValueError:
            pytest.fail(
                "La fonction n'a pas géré correctement l'erreur de suppression de collection"
            )


def test_local_environment(mock_chromadb_client):
    """
    Teste le comportement en environnement local.
    """
    with (
        patch.dict(os.environ, {"ENV": "local"}),
        patch("scripts.update_chroma.clone_or_update_repo"),
        patch("scripts.update_chroma.read_markdown_files"),
        patch("scripts.update_chroma.process_documents_for_chroma"),
        patch("scripts.update_chroma.chromadb.Client") as mock_client_class,
        patch("scripts.update_chroma.export_data") as mock_export,
    ):
        mock_client_class.return_value = mock_chromadb_client

        # Exécution de la fonction principale
        main()

        # Vérifier que l'export a été appelé avec les bons paramètres
        mock_export.assert_called_once_with(CHROMA_PERSIST_DIR, "chroma_db")


@pytest.mark.parametrize(
    "env_vars",
    [
        {},  # Aucune variable d'environnement
        {"ENV": "local", "REPO_URL": ""},  # URL de repo vide
        {"ENV": "prod", "S3_BUCKET_NAME": ""},  # Nom de bucket vide en prod
    ],
)
def test_missing_environment_variables(env_vars, mock_chromadb_client):
    """
    Teste le comportement avec des variables d'environnement manquantes.
    """
    with (
        patch.dict(os.environ, env_vars, clear=True),
        patch("scripts.update_chroma.clone_or_update_repo"),
        patch("scripts.update_chroma.chromadb.Client") as mock_client_class,
    ):
        mock_client_class.return_value = mock_chromadb_client

        # La fonction devrait utiliser les valeurs par défaut
        main()  # Ne devrait pas lever d'exception
