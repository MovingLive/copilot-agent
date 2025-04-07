"""
Tests unitaires pour le script update_faiss.py
"""

import json
import os
import unittest.mock as mock
from unittest.mock import patch, MagicMock
import numpy as np
import pytest
from moto import mock_aws
import faiss

from scripts.update_faiss import create_faiss_index, main, load_embedding_model
from app.services.embedding_service import EXPECTED_DIMENSION
from app.utils.git_utils import clone_or_update_repo
from app.utils.document_utils import read_relevant_files
from app.utils.vector_db_utils import process_files_for_faiss
from tests.conftest import MockSentenceTransformer

class MockSettings:
    """Mock des settings pour les tests."""
    REPO_DIR = "test_repo"
    SEGMENT_MAX_LENGTH = 1000
    TEMP_FAISS_DIR = "/tmp/test_faiss"
    FAISS_INDEX_FILE = "index.faiss"
    FAISS_METADATA_FILE = "metadata.json"
    S3_BUCKET_PREFIX = "faiss_index"
    S3_BUCKET_NAME = "test-bucket"
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    ENV = "test"  # Ajout de l'attribut ENV manquant


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """Fixture pour mocker les settings."""
    mock_settings = MockSettings()
    monkeypatch.setattr("scripts.update_faiss.settings", mock_settings)
    monkeypatch.setattr("app.services.faiss_service.settings", mock_settings)
    return mock_settings


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Fixture pour configurer les variables d'environnement de test."""
    with patch.dict(
        os.environ,
        {
            "ENV": "test",
            "REPO_DIR": MockSettings.REPO_DIR,
            "S3_BUCKET_NAME": MockSettings.S3_BUCKET_NAME,
        },
        clear=True
    ):
        yield

# Les fixtures pour les tests
@pytest.fixture
def mock_documents():
    """Fixture pour simuler les documents de test."""
    return [
        ("docs/test1.md", "Test document 1"),
        ("docs/test2.md", "Test document 2"),
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
def mock_embeddings():
    """Fixture pour simuler les embeddings. Retourne un seul embedding par défaut."""
    return np.zeros((1, EXPECTED_DIMENSION), dtype=np.float32)


# Patch séparé pour load_embedding_model pour s'assurer qu'il retourne notre mock cohérent
@pytest.fixture
def patch_load_embedding_model(mock_sentence_transformer):
    """Patch la fonction load_embedding_model pour qu'elle retourne notre mock"""
    with patch('sentence_transformers.SentenceTransformer', return_value=mock_sentence_transformer):
        yield mock_sentence_transformer


@pytest.fixture
def mock_load_embedding_model(monkeypatch):
    """Fixture qui patche load_embedding_model pour les tests spécifiques"""
    mock = MockSentenceTransformer()
    monkeypatch.setattr('scripts.update_faiss.load_embedding_model', lambda *args, **kwargs: mock)
    return mock


@pytest.fixture(autouse=True)
def mock_subprocess_run():
    """Mock subprocess.run pour éviter les vrais appels git."""
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        yield mock_run


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


def test_main_workflow(
    mock_documents, mock_processed_docs, mock_embeddings, mock_env_vars, patch_load_embedding_model,
    mock_subprocess_run
):
    """
    Teste le flux principal du script.
    """
    with (
        patch("scripts.update_faiss.get_repo_urls", return_value=["https://github.com/test/repo.git"]),
        patch("scripts.update_faiss.read_relevant_files", autospec=True) as mock_read,
        patch("scripts.update_faiss.process_files_for_faiss", autospec=True) as mock_process,
        patch("scripts.update_faiss.save_faiss_index", autospec=True) as mock_save,
        patch("scripts.update_faiss.export_data", autospec=True) as mock_export,
        patch("scripts.update_faiss.clone_multiple_repos", autospec=True) as mock_clone,
    ):
        # Configuration des mocks
        mock_clone.return_value = ["test_repo_path"]
        mock_read.return_value = mock_documents
        mock_process.return_value = mock_processed_docs
        patch_load_embedding_model.encode.return_value = mock_embeddings

        # Exécution de la fonction principale
        main()

        # Vérifications
        mock_clone.assert_called_once()
        mock_read.assert_called_once_with("test_repo_path")
        mock_process.assert_called_once_with(mock_documents, MockSettings.SEGMENT_MAX_LENGTH)
        mock_save.assert_called_once_with(mock.ANY, mock.ANY, MockSettings.TEMP_FAISS_DIR)
        mock_export.assert_called_once()

@mock_aws
def test_s3_export(mock_env_vars, mock_embeddings, patch_load_embedding_model, mock_subprocess_run):
    """
    Teste l'exportation des données vers S3.
    """
    import boto3

    # Configurer le mock S3
    s3 = boto3.client("s3", region_name="us-east-1")
    s3.create_bucket(Bucket="test-bucket")

    with (
        patch("scripts.update_faiss.get_repo_urls", return_value=["https://github.com/test/repo.git"]),
        patch("scripts.update_faiss.read_relevant_files", autospec=True) as mock_read,
        patch("scripts.update_faiss.process_files_for_faiss", autospec=True) as mock_process,
        patch("scripts.update_faiss.save_faiss_index", autospec=True) as mock_save,
        patch("scripts.update_faiss.export_data", autospec=True) as mock_export,
        patch("scripts.update_faiss.clone_multiple_repos", autospec=True) as mock_clone,
    ):
        # Configuration des mocks
        mock_clone.return_value = ["test_repo_path"]
        mock_read.return_value = [("test/test.md", "test content")]
        mock_process.return_value = [{"numeric_id": 1, "text": "test", "metadata": {}}]
        patch_load_embedding_model.encode.return_value = mock_embeddings

        # Exécution de la fonction principale
        main()

        # Vérifications
        mock_save.assert_called_once_with(mock.ANY, mock.ANY, MockSettings.TEMP_FAISS_DIR)
        mock_export.assert_called_once_with(MockSettings.TEMP_FAISS_DIR, MockSettings.S3_BUCKET_PREFIX)


def test_error_handling(mock_env_vars):
    """
    Teste la gestion des erreurs.
    """
    with (
        patch("scripts.update_faiss.get_repo_urls", return_value=["https://github.com/test/repo.git"]),
        patch("scripts.update_faiss.clone_multiple_repos", return_value=[]) as mock_clone_multiple,
        patch("scripts.update_faiss.logging.error") as mock_error,
        pytest.raises(SystemExit),
    ):
        # Exécution de la fonction principale
        main()

        # Vérifier que l'erreur a été journalisée
        mock_error.assert_called()


def test_empty_documents(mock_env_vars, mock_embeddings, patch_load_embedding_model, mock_subprocess_run):
    """
    Teste le comportement avec une liste de documents vide.
    """
    with (
        patch("scripts.update_faiss.get_repo_urls", return_value=["https://github.com/test/repo.git"]),
        patch("scripts.update_faiss.read_relevant_files", autospec=True) as mock_read,
        patch("scripts.update_faiss.process_files_for_faiss", autospec=True) as mock_process,
        patch("scripts.update_faiss.save_faiss_index", autospec=True) as mock_save,
        patch("scripts.update_faiss.export_data", autospec=True) as mock_export,
        patch("scripts.update_faiss.clone_multiple_repos", autospec=True) as mock_clone,
        pytest.raises(ValueError, match="La liste de documents est vide"),
    ):
        # Configuration des mocks
        mock_clone.return_value = ["test_repo_path"]
        mock_read.return_value = []
        mock_process.return_value = []
        empty_embeddings = np.array([]).reshape((0, EXPECTED_DIMENSION))
        patch_load_embedding_model.encode.return_value = empty_embeddings

        # Exécution de la fonction principale
        main()


def test_load_embedding_model_mock():
    """
    Teste que load_embedding_model est bien mocker et ne tente pas de télécharger le modèle.
    """
    # On s'attend à ce que cette fonction ne lève pas d'exception car elle est patchée
    # par la fixture block_huggingface_requests qui est autouse=True
    model = load_embedding_model()

    # Au lieu de vérifier le type exact, on vérifie que c'est bien un objet mock
    # qui possède les méthodes attendues
    assert hasattr(model, 'encode')
    assert hasattr(model, 'get_sentence_embedding_dimension')
    assert callable(model.encode)
    # Vérifions que la méthode encode retourne un array de la bonne dimension
    test_text = "Test sentence"
    result = model.encode(test_text)
    assert result.shape[1] == EXPECTED_DIMENSION


@pytest.mark.parametrize(
    "env_vars",
    [
        {},  # Aucune variable d'environnement
        {"ENV": "prod", "S3_BUCKET_NAME": ""},  # Nom de bucket vide en prod
    ],
)
def test_missing_environment_variables(
    env_vars, mock_embeddings, patch_load_embedding_model, mock_subprocess_run
):
    """
    Teste le comportement avec des variables d'environnement manquantes.
    """
    with (
        patch.dict(os.environ, env_vars, clear=True),
        patch("scripts.update_faiss.get_repo_urls", return_value=["https://github.com/test/repo.git"]),
        patch("scripts.update_faiss.read_relevant_files", autospec=True) as mock_read,
        patch("scripts.update_faiss.process_files_for_faiss", autospec=True) as mock_process,
        patch("scripts.update_faiss.save_faiss_index", autospec=True) as mock_save,
        patch("scripts.update_faiss.export_data", autospec=True) as mock_export,
        patch("scripts.update_faiss.clone_multiple_repos", autospec=True) as mock_clone,
    ):
        # Configuration des mocks
        mock_clone.return_value = ["test_repo_path"]
        mock_read.return_value = [("test/test.md", "test content")]
        mock_process.return_value = [{"numeric_id": 1, "text": "test", "metadata": {}}]
        patch_load_embedding_model.encode.return_value = mock_embeddings

        # La fonction devrait utiliser les valeurs par défaut
        main()

        # Vérifications
        mock_save.assert_called_once_with(mock.ANY, mock.ANY, mock.ANY)
        mock_export.assert_called_once_with(mock.ANY, mock.ANY)


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
        [{"numeric_id": 1, "text": "test"}],  # Manque metadata
        [{"text": "test", "metadata": {}}],  # Manque numeric_id
    ],
)
def test_invalid_document_format(bad_input, mock_sentence_transformer):
    """
    Teste la gestion des documents avec un format invalide.
    """
    with pytest.raises(ValueError):
        create_faiss_index(bad_input, mock_sentence_transformer)
