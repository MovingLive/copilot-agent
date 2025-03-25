"""
Tests unitaires pour le script update_faiss.py
"""

import json
import os
from unittest.mock import patch
import numpy as np
import pytest
from moto import mock_aws
import faiss

from app.core.config import settings
from scripts.update_faiss import create_faiss_index, main, save_faiss_index
from app.services.embedding_service import EXPECTED_DIMENSION

# Les fixtures pour les tests
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
    return np.zeros((2, EXPECTED_DIMENSION), dtype=np.float32)


def test_create_faiss_index(mock_processed_docs, mock_sentence_transformer):
    """
    Teste la création d'un index FAISS avec les documents traités.
    """
    # Configuration du mock pour qu'il retourne des embeddings de la bonne dimension
    mock_embeddings = np.zeros((len(mock_processed_docs), EXPECTED_DIMENSION), dtype=np.float32)
    mock_sentence_transformer.encode.reset_mock()
    mock_sentence_transformer.encode.return_value = mock_embeddings

    # Exécution de la fonction
    index, mapping = create_faiss_index(mock_processed_docs, mock_sentence_transformer)

    # Vérifications
    assert isinstance(index, faiss.IndexIDMap)
    assert len(mapping) == len(mock_processed_docs)
    assert all(str(doc["numeric_id"]) in mapping for doc in mock_processed_docs)

    # Vérification de l'appel à encode
    mock_sentence_transformer.encode.assert_called_once_with(
        [doc["text"] for doc in mock_processed_docs],
        show_progress_bar=True
    )


def test_save_faiss_index(tmp_path, mock_processed_docs):
    """
    Teste la sauvegarde de l'index FAISS et du mapping.
    """
    # Créer un index de test
    dimension = 128
    index = faiss.IndexIDMap(faiss.IndexFlatL2(dimension))
    # pylint: disable=E1120
    index.add_with_ids(
        np.random.rand(2, dimension).astype("float32"), np.array([1, 2], dtype=np.int64)
    )

    # Créer un mapping de test avec des clés str pour correspondre au JSON
    mapping = {str(doc["numeric_id"]): doc["metadata"] for doc in mock_processed_docs}

    # Sauvegarder l'index
    save_faiss_index(index, mapping, str(tmp_path))

    # Vérifier que les fichiers ont été créés
    assert os.path.exists(os.path.join(tmp_path, settings.FAISS_INDEX_FILE))
    assert os.path.exists(os.path.join(tmp_path, settings.FAISS_METADATA_FILE))

    # Vérifier le contenu du mapping
    with open(os.path.join(tmp_path, settings.FAISS_METADATA_FILE), "r", encoding="utf-8") as f:
        saved_mapping = json.load(f)
    assert saved_mapping == mapping


def test_main_workflow(
    mock_documents, mock_processed_docs, mock_embeddings, mock_env_vars, mock_sentence_transformer
):
    """
    Teste le flux principal du script.
    """
    with (
        patch("scripts.update_faiss.clone_or_update_repo") as mock_clone,
        patch("scripts.update_faiss.read_markdown_files") as mock_read,
        patch("scripts.update_faiss.process_documents_for_faiss") as mock_process,
        patch("scripts.update_faiss.save_faiss_index") as mock_save,
        patch("scripts.update_faiss.export_data") as mock_export,
        patch("sentence_transformers.SentenceTransformer", return_value=mock_sentence_transformer),
    ):
        # Configuration des mocks
        mock_clone.return_value = "test_repo_path"
        mock_read.return_value = mock_documents
        mock_process.return_value = mock_processed_docs
        mock_sentence_transformer.encode.return_value = mock_embeddings

        # Exécution de la fonction principale
        main()

        # Vérifications
        mock_clone.assert_called_once_with(settings.REPO_URL, settings.REPO_DIR)
        mock_read.assert_called_once_with("test_repo_path")
        mock_process.assert_called_once_with(mock_documents, settings.SEGMENT_MAX_LENGTH)
        mock_save.assert_called_once()
        mock_export.assert_called_once()


@mock_aws
def test_s3_export(mock_env_vars, mock_embeddings, mock_sentence_transformer):
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
        patch("scripts.update_faiss.save_faiss_index"),
        patch("scripts.update_faiss.export_data") as mock_export,
        patch("sentence_transformers.SentenceTransformer", return_value=mock_sentence_transformer),
    ):
        # Configuration des mocks
        mock_clone.return_value = "test_repo_path"
        mock_read.return_value = [{"text": "test"}]
        mock_process.return_value = [{"numeric_id": 1, "text": "test", "metadata": {}}]
        mock_sentence_transformer.encode.return_value = mock_embeddings

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


def test_empty_documents(mock_env_vars, mock_embeddings, mock_sentence_transformer):
    """
    Teste le comportement avec une liste de documents vide.
    """
    with (
        patch("scripts.update_faiss.clone_or_update_repo") as mock_clone,
        patch("scripts.update_faiss.read_markdown_files") as mock_read,
        patch("scripts.update_faiss.process_documents_for_faiss") as mock_process,
        patch("scripts.update_faiss.save_faiss_index") as mock_save,
        patch("scripts.update_faiss.export_data") as mock_export,
        patch("sentence_transformers.SentenceTransformer", return_value=mock_sentence_transformer),
        pytest.raises(ValueError, match="La liste de documents est vide"),
    ):
        # Configuration des mocks
        mock_clone.return_value = "test_repo_path"
        mock_read.return_value = []
        mock_process.return_value = []
        empty_embeddings = np.array([]).reshape((0, EXPECTED_DIMENSION))
        mock_sentence_transformer.encode.return_value = empty_embeddings

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
def test_missing_environment_variables(env_vars, mock_embeddings, mock_sentence_transformer):
    """
    Teste le comportement avec des variables d'environnement manquantes.
    """
    with (
        patch.dict(os.environ, env_vars, clear=True),
        patch("scripts.update_faiss.clone_or_update_repo") as mock_clone,
        patch("scripts.update_faiss.read_markdown_files") as mock_read,
        patch("scripts.update_faiss.process_documents_for_faiss") as mock_process,  # Capturé correctement
    ):
        # Configuration des mocks
        mock_clone.return_value = "test_repo_path"
        mock_read.return_value = [{"text": "test"}]
        mock_process.return_value = [{"numeric_id": 1, "text": "test", "metadata": {}}]
        mock_sentence_transformer.encode.reset_mock()  # Réinitialiser le mock
        mock_sentence_transformer.encode.return_value = mock_embeddings

        # La fonction devrait utiliser les valeurs par défaut
        main()


def test_embedding_dimension_consistency(mock_processed_docs, mock_sentence_transformer, mock_env_vars):
    """
    Teste la cohérence des dimensions des embeddings générés.
    """
    # Configurer deux séries d'embeddings de dimensions identiques
    embeddings1 = np.zeros((2, EXPECTED_DIMENSION), dtype=np.float32)
    embeddings2 = np.zeros((2, EXPECTED_DIMENSION), dtype=np.float32)

    mock_sentence_transformer.encode.side_effect = [embeddings1, embeddings2]

    # Créer deux index avec les mêmes documents
    index1, _ = create_faiss_index(mock_processed_docs, mock_sentence_transformer)
    index2, _ = create_faiss_index(mock_processed_docs, mock_sentence_transformer)

    # Vérifier que les dimensions sont identiques
    assert index1.d == index2.d == EXPECTED_DIMENSION


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


def test_large_document_batch(mock_sentence_transformer):
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

    # Réinitialiser le mock et configurer le retour
    mock_sentence_transformer.encode.reset_mock()
    mock_sentence_transformer.encode.side_effect = None  # Supprimer tout side_effect précédent
    mock_sentence_transformer.encode.return_value = np.zeros((1000, EXPECTED_DIMENSION), dtype=np.float32)

    # Créer l'index
    index, mapping = create_faiss_index(large_docs, mock_sentence_transformer)

    # Vérifications
    assert index.ntotal == 1000
    assert len(mapping) == 1000


def test_faiss_search_functionality(mock_processed_docs, mock_embeddings, mock_sentence_transformer):
    """
    Teste la fonctionnalité de recherche de l'index FAISS.
    """
    # Réinitialiser et configurer le mock
    mock_sentence_transformer.encode.reset_mock()
    mock_sentence_transformer.encode.side_effect = None
    mock_sentence_transformer.encode.return_value = np.zeros((len(mock_processed_docs), EXPECTED_DIMENSION), dtype=np.float32)

    # Créer l'index
    index, mapping = create_faiss_index(mock_processed_docs, mock_sentence_transformer)

    # Simuler une recherche
    query_embedding = np.zeros((1, EXPECTED_DIMENSION), dtype=np.float32)
    distances, labels = index.search(x=query_embedding, k=2)

    # Vérifications
    assert len(labels[0]) == 2  # Nombre de résultats
    assert len(distances[0]) == 2  # Nombre de distances
    assert all(str(i) in mapping for i in labels[0])  # IDs valides


@pytest.mark.parametrize(
    "bad_input",
    [
        None,
        [],
        [{"numeric_id": 1}],  # Manque text et metadata
        [{"numeric_id": 1, "text": "Test"}],  # Manque metadata
    ],
)
def test_create_faiss_index_invalid_input(bad_input, mock_sentence_transformer):
    """
    Teste la gestion des entrées invalides dans create_faiss_index.
    """
    with pytest.raises((ValueError, AttributeError)):
        create_faiss_index(bad_input, mock_sentence_transformer)
