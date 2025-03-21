"""
Tests unitaires pour le module main.
"""

import json
import os
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import boto3
import faiss
import numpy as np
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from moto import mock_aws

from app.main import (
    QueryRequest,
    app,
    call_copilot_llm,
    embed_text,
    load_faiss_index,
    retrieve_similar_documents,
)


@pytest.fixture
def test_client():
    """Fixture pour le client de test FastAPI."""
    return TestClient(app)


@pytest.fixture
def mock_faiss_index():
    """Fixture pour simuler un index FAISS."""
    dimension = 128
    index = faiss.IndexFlatL2(dimension)
    # Ajouter quelques vecteurs de test
    vectors = np.random.rand(5, dimension).astype("float32")
    index.add(vectors)
    return index


@pytest.fixture
def mock_document_store():
    """Fixture pour simuler le store de documents."""
    return [
        {"id": 1, "content": "Test document 1"},
        {"id": 2, "content": "Test document 2"},
        {"id": 3, "content": "Test document 3"},
        {"id": 4, "content": "Test document 4"},
        {"id": 5, "content": "Test document 5"},
    ]


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
    assert len(embedding) == 128
    # Vérifie que les valeurs sont entre 0 et 1
    assert all(0 <= x <= 1 for x in embedding)
    # Vérifie que le même texte produit le même embedding
    assert embed_text(text) == embedding


def test_load_faiss_index_local(mock_faiss_index):
    """
    Teste le chargement de l'index FAISS en environnement local.
    """
    with (
        patch("app.main.is_local_environment", return_value=True),
        patch("faiss.read_index", return_value=mock_faiss_index),
        patch("builtins.open", mock_open(read_data='[{"id": 1, "content": "test"}]')),
        patch("os.path.exists", return_value=True),
    ):
        load_faiss_index()
        from app.main import FAISS_INDEX, document_store

        assert FAISS_INDEX is not None
        assert isinstance(FAISS_INDEX, faiss.IndexFlatL2)
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

    with (
        patch("app.main.is_local_environment", return_value=False),
        patch("faiss.read_index") as mock_read_index,
        patch("boto3.client") as mock_boto3_client,
        patch("builtins.open", mock_open(read_data='[{"id": 1, "content": "test"}]')),
    ):
        # Configure le mock S3 pour simuler le téléchargement
        mock_s3 = MagicMock()
        mock_boto3_client.return_value = mock_s3

        # Configure le mock de l'index FAISS
        mock_index = MagicMock()
        mock_read_index.return_value = mock_index

        # Exécute la fonction
        load_faiss_index()
        from app.main import FAISS_INDEX, document_store

        # Vérifie que l'index a été chargé
        assert FAISS_INDEX is mock_index
        # Vérifie que le document store a été chargé
        assert len(document_store) == 1
        assert document_store[0] == {"id": 1, "content": "test"}


def test_retrieve_similar_documents(mock_faiss_index, mock_document_store):
    """
    Teste la récupération de documents similaires.
    """
    with (
        patch("app.main.FAISS_INDEX", mock_faiss_index),
        patch("app.main.document_store", mock_document_store),
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

        answer = call_copilot_llm("Test question", "Test context")

        assert answer == "Test answer"
        mock_post.assert_called_once()


@pytest.mark.asyncio
async def test_query_endpoint(test_client, mock_faiss_index, mock_document_store):
    """
    Teste l'endpoint /query de l'API.
    """
    with (
        patch("app.main.FAISS_INDEX", mock_faiss_index),
        patch("app.main.document_store", mock_document_store),
        patch("app.main.call_copilot_llm", return_value="Test answer"),
    ):
        response = test_client.post(
            "/query",
            json={"question": "Test question", "context": "Additional context"},
        )

        assert response.status_code == 200
        assert response.json() == {"answer": "Test answer"}


def test_call_copilot_llm_error():
    """
    Teste la gestion des erreurs lors de l'appel à l'API Copilot LLM.
    """
    with (
        patch("requests.post", side_effect=Exception("API Error")),
        pytest.raises(HTTPException) as exc_info,
    ):
        call_copilot_llm("Test question", "Test context")

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
    """
    with (
        patch("app.main.is_local_environment", return_value=True),
        patch("os.path.exists", return_value=False),
        patch("faiss.IndexFlatL2") as mock_index,
        patch("faiss.write_index"),
        patch("faiss.read_index"),
    ):
        load_faiss_index()
        mock_index.assert_called_once_with(128)


@pytest.mark.asyncio
async def test_query_endpoint_invalid_request(test_client):
    """
    Teste l'endpoint /query avec une requête invalide.
    """
    response = test_client.post("/query", json={})
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_query_endpoint_no_context(
    test_client, mock_faiss_index, mock_document_store
):
    """
    Teste l'endpoint /query sans contexte additionnel.
    """
    with (
        patch("app.main.FAISS_INDEX", mock_faiss_index),
        patch("app.main.document_store", mock_document_store),
        patch("app.main.call_copilot_llm", return_value="Test answer"),
    ):
        response = test_client.post(
            "/query",
            json={"question": "Test question"},
        )

        assert response.status_code == 200
        assert response.json() == {"answer": "Test answer"}


def test_embed_text_consistency():
    """
    Teste la cohérence des embeddings générés.
    """
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
        embedding2 = embed_text(text)

        # Vérifie que le même texte produit toujours le même embedding
        assert embedding1 == embedding2
        # Vérifie la dimension
        assert len(embedding1) == 128
        # Vérifie les valeurs
        assert all(0 <= x <= 1 for x in embedding1)


@mock_aws
def test_load_faiss_index_s3_error():
    """
    Teste la gestion des erreurs lors du chargement depuis S3.
    """
    with (
        patch("app.main.is_local_environment", return_value=False),
        patch("boto3.client") as mock_client,
        patch("faiss.read_index") as mock_read_index,
    ):
        mock_client.return_value.download_file.side_effect = Exception("S3 Error")
        mock_read_index.side_effect = Exception("Failed to read index")

        # Vérifie que la fonction gère l'erreur sans crash
        load_faiss_index()
        from app.main import FAISS_INDEX

        # L'index devrait être None en cas d'erreur
        assert FAISS_INDEX is None
