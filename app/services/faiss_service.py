"""Service de gestion des opérations FAISS.
Fournit les fonctionnalités de recherche vectorielle avec FAISS.
"""

import json
import logging
import time

import boto3
import faiss
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)

# Constantes
UPDATE_INTERVAL = 3600  # Intervalle de mise à jour en secondes
HTTP_500_ERROR = "Erreur interne du service FAISS"
HTTP_400_ERROR = "Requête invalide"

# Variables globales du module
_index: faiss.Index | None = None
_document_store: dict = {}


# Exceptions
class FAISSServiceError(Exception):
    """Exception de base pour les erreurs du service FAISS."""


class FAISSLoadError(FAISSServiceError):
    """Erreur lors du chargement de l'index FAISS."""


class FAISSSyncError(FAISSServiceError):
    """Erreur lors de la synchronisation avec S3."""


def _get_local_path(filename: str) -> str:
    """Construit le chemin local pour un fichier."""
    base_dir = settings.LOCAL_OUTPUT_DIR if settings.ENV == "local" else "/tmp"
    return f"{base_dir}/{filename}"


def _download_from_s3(faiss_path: str, metadata_path: str) -> None:
    """Télécharge les fichiers depuis S3."""
    try:
        s3_client = boto3.client("s3", region_name=settings.AWS_REGION)
        for file_info in [
            (settings.FAISS_INDEX_FILE, faiss_path),
            (settings.FAISS_METADATA_FILE, metadata_path),
        ]:
            s3_client.download_file(settings.S3_BUCKET_NAME, file_info[0], file_info[1])
    except (BotoCoreError, ClientError) as e:
        raise FAISSSyncError(f"Erreur S3: {str(e)}") from e


def _load_faiss_index(path: str) -> faiss.Index:
    """Charge l'index FAISS depuis un fichier."""
    try:
        index = faiss.read_index(path)
        if index.ntotal == 0:
            raise FAISSLoadError("L'index FAISS est vide")

        logger.info(
            "Index FAISS chargé: dimension=%d, nombre d'éléments=%d",
            index.d,
            index.ntotal,
        )
        return index
    except OSError as e:
        raise FAISSLoadError(f"Impossible de lire l'index FAISS: {str(e)}") from e


def _load_document_store(path: str) -> dict:
    """Charge le document store depuis un fichier JSON."""
    try:
        with open(path, encoding="utf-8") as f:
            store = json.load(f)
        if not store:
            raise FAISSLoadError("Le document store est vide")
        logger.info("Document store chargé avec %d entrées", len(store))
        return store
    except (OSError, json.JSONDecodeError) as e:
        raise FAISSLoadError(f"Impossible de lire le document store: {str(e)}") from e


def _handle_load_error(message: str, error: Exception) -> None:
    """Gestion uniforme des erreurs de chargement."""
    logger.error("%s: %s", message, error)
    global _index, _document_store
    _index = None
    _document_store = {}
    if isinstance(error, FAISSServiceError):
        raise HTTPException(status_code=500, detail=str(error)) from error
    raise HTTPException(status_code=500, detail=f"{message}: {str(error)}") from error


def load_index() -> None:
    """Charge l'index FAISS et le document store."""
    try:
        local_faiss_path = _get_local_path(settings.FAISS_INDEX_FILE)
        local_metadata_path = _get_local_path(settings.FAISS_METADATA_FILE)

        if settings.ENV != "local":
            _download_from_s3(local_faiss_path, local_metadata_path)

        global _index, _document_store
        _index = _load_faiss_index(local_faiss_path)
        _document_store = _load_document_store(local_metadata_path)

    except (OSError, json.JSONDecodeError) as e:
        _handle_load_error("Erreur lors du chargement des fichiers", e)
    except (BotoCoreError, ClientError) as e:
        _handle_load_error("Erreur lors de la synchronisation S3", e)
    except FAISSServiceError as e:
        _handle_load_error("Erreur du service FAISS", e)
    except Exception as e:
        _handle_load_error("Erreur inattendue", e)


def update_periodically() -> None:
    """Met à jour l'index FAISS périodiquement.
    Cette fonction est conçue pour être exécutée dans un thread séparé.
    """
    while True:
        try:
            logger.info("Mise à jour périodique de l'index FAISS...")
            load_index()
            logger.info("Mise à jour de l'index FAISS terminée")
        except Exception as e:
            logger.error("Erreur lors de la mise à jour périodique: %s", e)
        time.sleep(UPDATE_INTERVAL)


def _prepare_query_vector(query_vector: np.ndarray) -> np.ndarray:
    """Prépare le vecteur de requête avec la bonne dimension."""
    if _index is None:
        load_index()
        if _index is None:
            raise FAISSLoadError("Index FAISS non initialisé")

    if query_vector.shape[1] != _index.d:
        new_vector = np.zeros((1, _index.d), dtype="float32")
        min_dim = min(query_vector.shape[1], _index.d)
        new_vector[0, :min_dim] = query_vector[0, :min_dim]
        return new_vector

    return query_vector


def _search_in_index(query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Effectue la recherche dans l'index FAISS."""
    try:
        return _index.search(query_vector, k)
    except RuntimeError as e:
        raise FAISSServiceError(f"Erreur lors de la recherche FAISS: {str(e)}") from e


def _process_search_results(distances: np.ndarray, indices: np.ndarray) -> list[dict]:
    """Traite les résultats de la recherche FAISS."""
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < 0 or str(idx) not in _document_store:
            continue

        doc = _document_store[str(idx)]
        if not isinstance(doc, dict) or "content" not in doc:
            continue

        content = doc.get("content", "")
        if content and len(content) > settings.MIN_SEGMENT_LENGTH:
            results.append(
                {
                    "content": content,
                    "distance": float(distances[0][i]),
                    "metadata": {k: v for k, v in doc.items() if k != "content"},
                }
            )

    return results


def search_similar(query_vector: np.ndarray, k: int = 5) -> list[dict]:
    """Recherche les k documents les plus similaires.

    Args:
        query_vector: Vecteur de requête
        k: Nombre de résultats à retourner

    Returns:
        list[dict]: Liste des documents similaires avec leurs métadonnées

    Raises:
        HTTPException: En cas d'erreur
    """
    try:
        prepared_vector = _prepare_query_vector(query_vector)
        distances, indices = _search_in_index(prepared_vector, k)
        return _process_search_results(distances, indices)

    except ValueError as ve:
        raise HTTPException(
            status_code=400, detail=f"{HTTP_400_ERROR}: {str(ve)}"
        ) from ve
    except FAISSServiceError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        logger.error("Erreur lors de la recherche: %s", e)
        raise HTTPException(status_code=500, detail=HTTP_500_ERROR) from e
