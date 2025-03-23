"""
Tests unitaires pour le module main.
"""

import os
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import boto3
import faiss
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from moto import mock_aws

from scripts.update_faiss import FAISS_INDEX_FILE, FAISS_METADATA_FILE
from app.main import (
    app,
    call_copilot_llm,
    embed_text,
    generate_query_vector,
    ensure_compatible_dimensions,
    search_faiss_index,
    extract_documents_from_indices,
    load_faiss_index,
    retrieve_similar_documents,
    get_local_faiss_path,
    load_and_validate_faiss_index,
    get_metadata_path,
    load_and_validate_document_store,
)


@pytest.fixture
def test_client():
    """Fixture pour le client de test FastAPI."""
    return TestClient(app)


@pytest.fixture
def mock_faiss_index():
    """Fixture pour simuler un index FAISS."""
    dimension = 384  # Mise à jour de la dimension à 384
    index = faiss.IndexFlatL2(dimension)
    # Ajouter quelques vecteurs de test
    vectors = np.random.rand(5, dimension).astype("float32")
    index.add(vectors)
    return index


@pytest.fixture
def mock_document_store():
    """Fixture pour simuler le store de documents."""
    # Conversion en dictionnaire avec ID comme clés (comme l'implémentation actuelle)
    return {
        "0": {"id": 1, "content": "Test document 1"},
        "1": {"id": 2, "content": "Test document 2"},
        "2": {"id": 3, "content": "Test document 3"},
        "3": {"id": 4, "content": "Test document 4"},
        "4": {"id": 5, "content": "Test document 5"},
    }


@pytest.fixture
def mock_query_vector():
    """Fixture pour simuler un vecteur de requête."""
    return np.random.rand(1, 384).astype("float32")


@pytest.fixture(autouse=True)
def mock_env_vars():
    """Fixture pour configurer les variables d'environnement de test."""
    with patch.dict(
        os.environ,
        {
            "ENV": "test",
            "S3_BUCKET_NAME": "test-bucket",
            "AWS_REGION": "ca-central-1",
            "COPILOT_API_URL": "https://test-api.com",
            "COPILOT_TOKEN": "test-token",
        },
    ):
        yield


def test_root_endpoint(test_client):
    """
    Teste l'endpoint racine de l'API.
    """
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Bienvenue dans l'API Copilot LLM!"}


def test_embed_text():
    """
    Teste la fonction d'embedding de texte.
    """
    text = "Test text"
    embedding = embed_text(text)

    # Vérifie que l'embedding a la bonne dimension
    assert len(embedding) == 384  # Mise à jour pour correspondre à la valeur actuelle
    # Vérifie que les valeurs sont entre 0 et 1
    assert all(0 <= x <= 1 for x in embedding)
    # Vérifie que le même texte produit le même embedding
    assert embed_text(text) == embedding


def test_generate_query_vector():
    """
    Teste la génération du vecteur de requête.
    """
    with patch("app.main.embed_text", return_value=np.random.rand(384).tolist()):
        query = "test query"
        query_vector = generate_query_vector(query)

        # Vérifie le type et la forme du vecteur
        assert isinstance(query_vector, np.ndarray)
        assert query_vector.shape[1] == 384
        assert query_vector.dtype == "float32"


def test_ensure_compatible_dimensions(mock_faiss_index, mock_query_vector):
    """
    Teste l'ajustement des dimensions du vecteur de requête.
    """
    # Test avec des dimensions compatibles
    compatible_vector = mock_query_vector
    adjusted_vector = ensure_compatible_dimensions(compatible_vector, mock_faiss_index)
    assert adjusted_vector.shape[1] == mock_faiss_index.d

    # Test avec des dimensions incompatibles
    incompatible_vector = np.random.rand(1, 128).astype("float32")  # Dimension différente
    adjusted_vector = ensure_compatible_dimensions(incompatible_vector, mock_faiss_index)
    assert adjusted_vector.shape[1] == mock_faiss_index.d


def test_search_faiss_index(mock_faiss_index, mock_query_vector):
    """
    Teste la recherche dans l'index FAISS.
    """
    # Test d'une recherche réussie
    distances, indices = search_faiss_index(mock_query_vector, mock_faiss_index, k=3)

    assert isinstance(distances, np.ndarray)
    assert isinstance(indices, np.ndarray)
    assert distances.shape[0] == 1  # Une seule requête
    assert indices.shape[0] == 1    # Une seule requête
    assert indices.shape[1] == 3    # k=3 résultats

    # Test de la gestion des erreurs
    with patch.object(mock_faiss_index, "search", side_effect=AssertionError("Test error")):
        with pytest.raises(HTTPException) as exc_info:
            search_faiss_index(mock_query_vector, mock_faiss_index, k=3)

        assert exc_info.value.status_code == 500
        assert "Erreur d'assertion lors de la recherche FAISS" in str(exc_info.value.detail)


def test_extract_documents_from_indices(mock_document_store):
    """
    Teste l'extraction des documents à partir des indices.
    """
    indices = np.array([[0, 1, 2]])
    distances = np.array([[0.1, 0.2, 0.3]])

    docs = extract_documents_from_indices(indices, distances, mock_document_store)

    assert len(docs) == 3
    assert all(isinstance(doc, dict) for doc in docs)
    assert all("content" in doc for doc in docs)

    # Test avec un indice inexistant
    indices = np.array([[0, 99, 2]])  # 99 n'existe pas
    docs = extract_documents_from_indices(indices, distances, mock_document_store)
    assert len(docs) == 2  # Seulement 2 documents valides


def test_get_local_faiss_path():
    """
    Teste la détermination du chemin local pour l'index FAISS.
    """
    # Test en environnement local avec fichier existant
    with patch("app.main.is_local_environment", return_value=True), \
         patch("os.path.exists", return_value=True):
        path = get_local_faiss_path()
        assert path.endswith(FAISS_INDEX_FILE)
        assert "/output/" in path

    # Test en environnement local sans fichier existant
    with patch("app.main.is_local_environment", return_value=True), \
         patch("os.path.exists", return_value=False), \
         patch("faiss.write_index"):
        path = get_local_faiss_path()
        assert path.startswith("/tmp/")
        assert path.endswith(FAISS_INDEX_FILE)

    # Test en environnement distant (S3)
    with patch("app.main.is_local_environment", return_value=False), \
         patch("boto3.client") as mock_boto3:
        path = get_local_faiss_path()
        assert path.startswith("/tmp/")
        assert path.endswith(FAISS_INDEX_FILE)
        assert mock_boto3.called


def test_load_and_validate_faiss_index(mock_faiss_index):
    """
    Teste le chargement et la validation de l'index FAISS.
    """
    with patch("faiss.read_index", return_value=mock_faiss_index):
        index = load_and_validate_faiss_index("/path/to/index.faiss")
        assert index is mock_faiss_index
        assert index.ntotal == 5  # 5 vecteurs ajoutés dans la fixture


def test_get_metadata_path():
    """
    Teste la détermination du chemin des métadonnées.
    """
    # Test en environnement local avec fichier existant
    with patch("app.main.is_local_environment", return_value=True), \
         patch("os.path.exists", return_value=True):
        path = get_metadata_path()
        assert path.endswith(FAISS_METADATA_FILE)

    # Test en environnement local sans fichier existant
    with patch("app.main.is_local_environment", return_value=True), \
         patch("os.path.exists", return_value=False):
        with pytest.raises(FileNotFoundError):
            get_metadata_path()

    # Test en environnement distant (S3)
    with patch("app.main.is_local_environment", return_value=False), \
         patch("boto3.client") as mock_boto3:
        path = get_metadata_path()
        assert path.startswith("/tmp/")
        assert path.endswith(FAISS_METADATA_FILE)
        assert mock_boto3.called


def test_load_and_validate_document_store():
    """
    Teste le chargement et la validation du document store.
    """
    test_data = '{"0": {"id": 1, "content": "test"}}'
    with patch("builtins.open", mock_open(read_data=test_data)):
        doc_store = load_and_validate_document_store("/path/to/metadata.json")
        assert isinstance(doc_store, dict)
        assert len(doc_store) == 1
        assert "0" in doc_store
        assert doc_store["0"]["content"] == "test"


def test_load_faiss_index_local(mock_faiss_index):
    """
    Teste le chargement de l'index FAISS en environnement local.
    """
    with (
        patch("app.main.get_local_faiss_path", return_value="/test/path/index.faiss"),
        patch("app.main.load_and_validate_faiss_index", return_value=mock_faiss_index),
        patch("app.main.get_metadata_path", return_value="/test/path/metadata.json"),
        patch("app.main.load_and_validate_document_store", return_value={"0": {"content": "test"}}),
    ):
        load_faiss_index()
        from app.main import FAISS_INDEX, document_store

        assert FAISS_INDEX is mock_faiss_index
        assert isinstance(document_store, dict)
        assert len(document_store) == 1


@mock_aws
def test_load_faiss_index_s3():
    """
    Teste le chargement de l'index FAISS depuis S3.
    """
    # Configurer le mock S3
    s3 = boto3.client("s3", region_name="ca-central-1")
    s3.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "ca-central-1"},
    )

    mock_index = MagicMock()
    mock_doc_store = {"0": {"id": 1, "content": "test"}}

    with (
        patch("app.main.get_local_faiss_path", return_value="/tmp/index.faiss"),
        patch("app.main.load_and_validate_faiss_index", return_value=mock_index),
        patch("app.main.get_metadata_path", return_value="/tmp/metadata.json"),
        patch("app.main.load_and_validate_document_store", return_value=mock_doc_store),
    ):
        # Exécute la fonction
        load_faiss_index()
        from app.main import FAISS_INDEX, document_store

        # Vérifie que l'index a été chargé
        assert FAISS_INDEX is mock_index
        # Vérifie que le document store a été chargé
        assert document_store is mock_doc_store


def test_retrieve_similar_documents(mock_faiss_index, mock_document_store, mock_query_vector):
    """
    Teste la récupération de documents similaires.
    """
    with (
        patch("app.main.FAISS_INDEX", mock_faiss_index),
        patch("app.main.document_store", mock_document_store),
        patch("app.main.generate_query_vector", return_value=mock_query_vector),
        patch("app.main.ensure_compatible_dimensions", return_value=mock_query_vector),
        patch("app.main.search_faiss_index", return_value=(np.array([[0.1, 0.2, 0.3]]), np.array([[0, 1, 2]]))),
        patch("app.main.extract_documents_from_indices", return_value=[
            {"content": "Test document 1"},
            {"content": "Test document 2"},
            {"content": "Test document 3"}
        ]),
    ):
        docs = retrieve_similar_documents("test query", k=3)

        assert len(docs) == 3
        assert all(isinstance(doc, dict) for doc in docs)
        assert all("content" in doc for doc in docs)


def test_call_copilot_llm():
    """
    Teste l'appel à l'API Copilot LLM.
    """
    mock_response = {"choices": [{"message": {"content": "Test answer"}}]}

    with patch("requests.post") as mock_post:
        mock_post.return_value.json.return_value = mock_response
        mock_post.return_value.status_code = 200
        mock_post.return_value.ok = True

        answer = call_copilot_llm("Test question", "Test context", "test-auth-token")

        assert answer == "Test answer"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_query_endpoint(test_client, mock_faiss_index, mock_document_store):
    """
    Teste l'endpoint principal de l'API pour les requêtes.
    """
    # Patch directement la fonction handle_copilot_query pour éviter les problèmes d'authentification
    with patch("app.main.handle_copilot_query", new_callable=AsyncMock) as mock_handler:
        # Configurer le mock pour renvoyer une réponse simulée
        mock_stream = AsyncMock()
        async def fake_generator():
            yield b'{"choices": [{"message": {"content": "Test answer"}}]}'

        mock_stream.return_value = StreamingResponse(fake_generator())
        mock_handler.return_value = mock_stream.return_value

        # Exécution du test
        response = test_client.post(
            "/",
            headers={"x-github-token": "valid-test-token"},
            json={
                "messages": [{"content": "Test question", "role": "user"}],
                "copilot_references": "Additional context"
            },
        )

        # Vérifier que le mock a été appelé
        assert mock_handler.called
        # Vérifier que la réponse est celle attendue
        assert response.status_code == 200


def test_call_copilot_llm_error():
    """
    Teste la gestion des erreurs lors de l'appel à l'API Copilot LLM.
    """
    with (
        patch("requests.post", side_effect=Exception("API Error")),
        pytest.raises(HTTPException) as exc_info,
    ):
        call_copilot_llm("Test question", "Test context", "test-auth-token")

        assert exc_info.value.status_code == 500
        assert "Erreur lors de l'appel au service Copilot LLM" in str(
            exc_info.value.detail
        )


def test_retrieve_similar_documents_empty_index():
    """
    Teste la récupération de documents avec un index vide.
    """
    with patch("app.main.FAISS_INDEX", None):
        docs = retrieve_similar_documents("test query", k=3)
        assert len(docs) == 0


def test_load_faiss_index_local_no_files():
    """
    Teste le chargement de l'index FAISS quand les fichiers n'existent pas.
    Un index vide doit être créé avec la dimension du modèle all-MiniLM-L6-v2 (384).
    """
    with (
        patch("app.main.get_local_faiss_path") as mock_get_path,
        patch("faiss.IndexFlatL2") as mock_index,
        patch("faiss.write_index"),
        patch("app.main.load_and_validate_faiss_index") as mock_load,
        patch("app.main.get_metadata_path", side_effect=FileNotFoundError("No metadata")),
    ):
        mock_get_path.return_value = "/tmp/index.faiss"
        mock_index.return_value = MagicMock()
        mock_load.return_value = mock_index.return_value

        load_faiss_index()
        from app.main import FAISS_INDEX, document_store

        assert FAISS_INDEX is mock_index.return_value
        assert document_store == []


@pytest.mark.asyncio
async def test_query_endpoint_invalid_request(test_client):
    """
    Teste l'endpoint principal avec une requête invalide.
    """
    # Pour les requêtes sans token, on doit obtenir 401 et non 422
    response = test_client.post("/", json={})
    assert response.status_code == 401  # Unauthorized error, pas de token d'authentification


@pytest.mark.asyncio
async def test_query_endpoint_no_context(test_client, mock_faiss_index, mock_document_store):
    """
    Teste l'endpoint principal sans contexte additionnel.
    """
    # Patch directement la fonction handle_copilot_query pour éviter les problèmes d'authentification
    with patch("app.main.handle_copilot_query", new_callable=AsyncMock) as mock_handler:
        # Configurer le mock pour renvoyer une réponse simulée
        mock_stream = AsyncMock()
        async def fake_generator():
            yield b'{"choices": [{"message": {"content": "Test answer without context"}}]}'

        mock_stream.return_value = StreamingResponse(fake_generator())
        mock_handler.return_value = mock_stream.return_value

        # Exécution du test
        response = test_client.post(
            "/",
            headers={"x-github-token": "test-token"},
            json={
                "messages": [{"content": "Test question", "role": "user"}],
                # Pas de contexte additionnel ici
            },
        )

        # Vérifier que le mock a été appelé
        assert mock_handler.called
        assert response.status_code == 200


def test_embed_text_consistency():
    """
    Teste la cohérence des embeddings générés.
    """
    # Patch la fonction embed_text pour garantir un vecteur de dimension 384
    with patch("app.main.embed_text", side_effect=lambda text: np.random.rand(384).tolist()):
        # Teste avec différents types de textes
        texts = [
            "Test text",
            "Another test",
            "",  # Texte vide
            "Test text with numbers 123",
            "Test text with special chars !@#",
            "Very long text " * 100,  # Test avec un texte très long
        ]

        for text in texts:
            embedding1 = embed_text(text)
            embedding2 = embed_text(text)  # Même seed, même texte → même embedding

            # Vérifie que le même texte produit toujours le même embedding
            assert embedding1 == embedding2
            # Vérifie la dimension
            assert len(embedding1) == 384
            # Vérifie les valeurs
            assert all(0 <= x <= 1 for x in embedding1)


def test_generate_query_vector_empty_query():
    """
    Teste la gestion des erreurs avec une requête vide.
    """
    with pytest.raises(ValueError) as exc_info:
        generate_query_vector("")
    assert "Query cannot be empty" in str(exc_info.value)


@mock_aws
def test_load_faiss_index_s3_error():
    """
    Teste la gestion des erreurs lors du chargement depuis S3.
    """
    with (
        patch("app.main.get_local_faiss_path", side_effect=Exception("S3 Error")),
    ):
        # Vérifie que la fonction gère l'erreur sans crash
        load_faiss_index()
        from app.main import FAISS_INDEX

        # L'index devrait être None en cas d'erreur
        assert FAISS_INDEX is None
