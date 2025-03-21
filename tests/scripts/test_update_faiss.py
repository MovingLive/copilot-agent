"""
Tests unitaires pour le script update_faiss.py
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import faiss
import numpy as np
import pytest
from moto import mock_aws
from sentence_transformers import SentenceTransformer

from scripts.update_faiss import (
    FAISS_INDEX_FILE,
    FAISS_METADATA_FILE,
    FAISS_PERSIST_DIR,
    REPO_DIR,
    REPO_URL,
    SEGMENT_MAX_LENGTH,
    create_faiss_index,
    main,
    save_faiss_index,
)


@pytest.fixture
def mock_documents():
    """Fixture pour simuler les documents de test."""
    return [
        {"id": "doc1", "content": "Test document 1", "path": "docs/test1.md"},
        {"id": "doc2", "content": "Test document 2", "path": "docs/test2.md"},
    ]


@pytest.fixture
def mock_processed_docs():
    """Fixture pour simuler les documents traités."""
    return [
        {
            "numeric_id": 1,
            "text": "Test document 1",
            "metadata": {"source": "docs/test1.md", "segment": 0},
        },
        {
            "numeric_id": 2,
            "text": "Test document 2",
            "metadata": {"source": "docs/test2.md", "segment": 0},
        },
    ]


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


@pytest.fixture
def mock_embeddings():
    """Fixture pour simuler les embeddings."""
    return np.random.rand(2, 128).astype("float32")


def test_create_faiss_index(mock_processed_docs, mock_embeddings):
    """
    Teste la création d'un index FAISS avec les documents traités.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_model:
        # Configuration du mock
        model_instance = MagicMock()
        model_instance.encode.return_value = mock_embeddings
        mock_model.return_value = model_instance

        # Exécution de la fonction
        index, mapping = create_faiss_index(mock_processed_docs, mock_model())

        # Vérifications
        assert isinstance(index, faiss.IndexIDMap)
        assert len(mapping) == len(mock_processed_docs)
        assert all(doc["numeric_id"] in mapping for doc in mock_processed_docs)
        model_instance.encode.assert_called_once()


def test_save_faiss_index(tmp_path, mock_processed_docs):
    """
    Teste la sauvegarde de l'index FAISS et du mapping.
    """
    # Créer un index de test
    dimension = 128
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    index.add_with_ids(
        np.random.rand(2, dimension).astype("float32"), np.array([1, 2], dtype=np.int64)
    )

    # Créer un mapping de test avec des clés str pour correspondre au JSON
    mapping = {str(doc["numeric_id"]): doc["metadata"] for doc in mock_processed_docs}

    # Sauvegarder l'index
    save_faiss_index(index, mapping, str(tmp_path))

    # Vérifier que les fichiers ont été créés
    assert os.path.exists(os.path.join(tmp_path, FAISS_INDEX_FILE))
    assert os.path.exists(os.path.join(tmp_path, FAISS_METADATA_FILE))

    # Vérifier le contenu du mapping
    with open(os.path.join(tmp_path, FAISS_METADATA_FILE), "r") as f:
        saved_mapping = json.load(f)
    assert saved_mapping == mapping


def test_main_workflow(
    mock_documents, mock_processed_docs, mock_embeddings, mock_env_vars
):
    """
    Teste le flux principal du script.
    """
    with (
        patch("scripts.update_faiss.clone_or_update_repo") as mock_clone,
        patch("scripts.update_faiss.read_markdown_files") as mock_read,
        patch("scripts.update_faiss.process_documents_for_faiss") as mock_process,
        patch("sentence_transformers.SentenceTransformer") as mock_model,
        patch("scripts.update_faiss.save_faiss_index") as mock_save,
        patch("scripts.update_faiss.export_data") as mock_export,
    ):
        # Configuration des mocks
        mock_clone.return_value = "test_repo_path"
        mock_read.return_value = mock_documents
        mock_process.return_value = mock_processed_docs
        model_instance = MagicMock()
        model_instance.encode.return_value = mock_embeddings
        mock_model.return_value = model_instance

        # Exécution de la fonction principale
        main()

        # Vérifications
        mock_clone.assert_called_once_with(REPO_URL, REPO_DIR)
        mock_read.assert_called_once_with("test_repo_path")
        mock_process.assert_called_once_with(mock_documents, SEGMENT_MAX_LENGTH)
        mock_save.assert_called_once()
        mock_export.assert_called_once()


@mock_aws
def test_s3_export(mock_env_vars, mock_embeddings):
    """
    Teste l'exportation des données vers S3.
    """
    import boto3

    # Configurer le mock S3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")

    with (
        patch("scripts.update_faiss.clone_or_update_repo") as mock_clone,
        patch("scripts.update_faiss.read_markdown_files") as mock_read,
        patch("scripts.update_faiss.process_documents_for_faiss") as mock_process,
        patch("sentence_transformers.SentenceTransformer") as mock_model,
        patch("scripts.update_faiss.save_faiss_index"),
        patch("scripts.update_faiss.export_data") as mock_export,
    ):
        # Configuration des mocks
        mock_clone.return_value = "test_repo_path"
        mock_read.return_value = [{"text": "test"}]
        mock_process.return_value = [{"numeric_id": 1, "text": "test", "metadata": {}}]
        model_instance = MagicMock()
        model_instance.encode.return_value = mock_embeddings
        mock_model.return_value = model_instance

        # Exécution de la fonction principale
        main()

        # Vérification que l'export a été appelé
        mock_export.assert_called_once()


def test_error_handling(mock_env_vars):
    """
    Teste la gestion des erreurs.
    """
    with (
        patch("scripts.update_faiss.clone_or_update_repo") as mock_clone,
        patch("scripts.update_faiss.logging.error") as mock_error,
        pytest.raises(Exception),
    ):
        # Simuler une erreur lors du clonage
        mock_clone.side_effect = Exception("Test error")

        # Exécution de la fonction principale
        main()

        # Vérifier que l'erreur a été journalisée
        mock_error.assert_called()


def test_empty_documents(mock_env_vars, mock_embeddings):
    """
    Teste le comportement avec une liste de documents vide.
    """
    with (
        patch("scripts.update_faiss.clone_or_update_repo") as mock_clone,
        patch("scripts.update_faiss.read_markdown_files") as mock_read,
        patch("scripts.update_faiss.process_documents_for_faiss") as mock_process,
        patch("sentence_transformers.SentenceTransformer") as mock_model,
        patch("scripts.update_faiss.save_faiss_index") as mock_save,
        patch("scripts.update_faiss.export_data") as mock_export,
        pytest.raises(ValueError, match="La liste de documents est vide"),
    ):
        # Configuration des mocks
        mock_clone.return_value = "test_repo_path"
        mock_read.return_value = []
        mock_process.return_value = []
        model_instance = MagicMock()
        model_instance.encode.return_value = np.array([]).reshape((0, 128))
        mock_model.return_value = model_instance

        # Exécution de la fonction principale
        main()


@pytest.mark.parametrize(
    "env_vars",
    [
        {},  # Aucune variable d'environnement
        {"ENV": "local", "REPO_URL": ""},  # URL de repo vide
        {"ENV": "prod", "S3_BUCKET_NAME": ""},  # Nom de bucket vide en prod
    ],
)
def test_missing_environment_variables(env_vars, mock_embeddings):
    """
    Teste le comportement avec des variables d'environnement manquantes.
    """
    with (
        patch.dict(os.environ, env_vars, clear=True),
        patch("scripts.update_faiss.clone_or_update_repo") as mock_clone,
        patch("scripts.update_faiss.read_markdown_files") as mock_read,
        patch("scripts.update_faiss.process_documents_for_faiss") as mock_process,
        patch("sentence_transformers.SentenceTransformer") as mock_model,
    ):
        # Configuration des mocks
        mock_clone.return_value = "test_repo_path"
        mock_read.return_value = [{"text": "test"}]
        mock_process.return_value = [{"numeric_id": 1, "text": "test", "metadata": {}}]
        model_instance = MagicMock()
        model_instance.encode.return_value = mock_embeddings
        mock_model.return_value = model_instance

        # La fonction devrait utiliser les valeurs par défaut
        main()


def test_embedding_dimension_consistency(mock_processed_docs):
    """
    Teste la cohérence des dimensions des embeddings générés.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_model:
        # Configurer deux séries d'embeddings de dimensions différentes
        embeddings1 = np.random.rand(2, 128).astype("float32")
        embeddings2 = np.random.rand(2, 128).astype("float32")

        model_instance = MagicMock()
        model_instance.encode.side_effect = [embeddings1, embeddings2]
        mock_model.return_value = model_instance

        # Créer deux index avec les mêmes documents
        index1, _ = create_faiss_index(mock_processed_docs, mock_model())
        index2, _ = create_faiss_index(mock_processed_docs, mock_model())

        # Vérifier que les dimensions sont identiques
        assert index1.d == index2.d == 128


def test_save_faiss_index_file_permissions(tmp_path, mock_processed_docs):
    """
    Teste la gestion des erreurs de permissions lors de la sauvegarde.
    """
    # Créer un index de test
    dimension = 128
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    mapping = {doc["numeric_id"]: doc["metadata"] for doc in mock_processed_docs}

    with (
        patch("builtins.open", side_effect=PermissionError),
        pytest.raises(PermissionError),
    ):
        save_faiss_index(index, mapping, str(tmp_path))


def test_large_document_batch():
    """
    Teste le traitement d'un grand nombre de documents.
    """
    # Créer un grand nombre de documents
    large_docs = [
        {
            "numeric_id": i,
            "text": f"Test document {i}",
            "metadata": {"source": f"docs/test{i}.md", "segment": 0},
        }
        for i in range(1000)
    ]

    with patch("sentence_transformers.SentenceTransformer") as mock_model:
        # Simuler des embeddings pour tous les documents
        embeddings = np.random.rand(1000, 128).astype("float32")
        model_instance = MagicMock()
        model_instance.encode.return_value = embeddings
        mock_model.return_value = model_instance

        # Créer l'index
        index, mapping = create_faiss_index(large_docs, mock_model())

        # Vérifications
        assert index.ntotal == 1000
        assert len(mapping) == 1000


def test_faiss_search_functionality(mock_processed_docs, mock_embeddings):
    """
    Teste la fonctionnalité de recherche de l'index FAISS.
    """
    with patch("sentence_transformers.SentenceTransformer") as mock_model:
        model_instance = MagicMock()
        model_instance.encode.return_value = mock_embeddings
        mock_model.return_value = model_instance

        # Créer l'index
        index, mapping = create_faiss_index(mock_processed_docs, mock_model())

        # Simuler une recherche
        query_embedding = np.random.rand(1, 128).astype("float32")
        D, I = index.search(query_embedding, k=2)

        # Vérifications
        assert len(I[0]) == 2  # Nombre de résultats
        assert len(D[0]) == 2  # Nombre de distances
        assert all(i in mapping for i in I[0])  # IDs valides


@pytest.mark.parametrize(
    "bad_input",
    [
        None,
        [],
        [{"numeric_id": 1}],  # Manque text et metadata
        [{"numeric_id": 1, "text": "Test"}],  # Manque metadata
    ],
)
def test_create_faiss_index_invalid_input(bad_input):
    """
    Teste la gestion des entrées invalides dans create_faiss_index.
    """
    with (
        patch("sentence_transformers.SentenceTransformer") as mock_model,
        pytest.raises((ValueError, AttributeError)),
    ):
        model_instance = MagicMock()
        mock_model.return_value = model_instance
        create_faiss_index(bad_input, mock_model())
