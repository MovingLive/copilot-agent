"""Service de gestion des opérations FAISS.

Fournit les fonctionnalités de recherche vectorielle avec FAISS.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import faiss
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings
from app.services.embedding_service import generate_query_vector
from app.utils.export_utils import is_local_environment

logger = logging.getLogger(__name__)

# Constantes
UPDATE_INTERVAL = 3600  # Intervalle de mise à jour en secondes
MIN_CONTENT_LENGTH = 10  # Longueur minimale de contenu à retourner
HTTP_500_ERROR = "Erreur interne du service FAISS"
HTTP_400_ERROR = "Requête invalide"


# Exceptions personnalisées
class FAISSServiceError(Exception):
    """Exception de base pour les erreurs du service FAISS."""


class FAISSLoadError(FAISSServiceError):
    """Erreur lors du chargement de l'index FAISS."""


class FAISSSyncError(FAISSServiceError):
    """Erreur lors de la synchronisation avec S3."""


@dataclass
class FAISSState:
    """État du service FAISS."""

    index: faiss.Index | None = None
    document_store: dict[str, Any] = None

    def __post_init__(self):
        """Initialise l'état avec des valeurs par défaut."""
        if self.document_store is None:
            self.document_store = {}


# Instance unique de l'état
_state = FAISSState()


def _get_local_path(filename: str) -> str:
    """Construit le chemin local pour un fichier."""
    base_dir = settings.LOCAL_OUTPUT_DIR if is_local_environment() else "/tmp"
    return str(Path(base_dir) / filename)


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


def _load_document_store(path: str) -> dict[str, Any]:
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


def load_index() -> tuple[faiss.Index | None, dict[str, Any]]:
    """Charge l'index FAISS et le document store."""
    try:
        local_faiss_path = _get_local_path(settings.FAISS_INDEX_FILE)
        local_metadata_path = _get_local_path(settings.FAISS_METADATA_FILE)

        if not is_local_environment():
            _download_from_s3(local_faiss_path, local_metadata_path)

        index = _load_faiss_index(local_faiss_path)
        doc_store = _load_document_store(local_metadata_path)
        return index, doc_store

    except (
        FAISSServiceError,
        OSError,
        json.JSONDecodeError,
        BotoCoreError,
        ClientError,
    ) as e:
        logger.error("Erreur lors du chargement: %s", e)
        return None, {}


async def update_periodically() -> None:
    """Met à jour l'index FAISS périodiquement.

    Cette fonction est conçue pour être exécutée dans un thread séparé.
    """
    while True:
        try:
            logger.info("Mise à jour périodique de l'index FAISS...")
            index, doc_store = load_index()
            _state.index = index
            _state.document_store = doc_store
            logger.info("Mise à jour de l'index FAISS terminée")
        except (FAISSServiceError, OSError, json.JSONDecodeError) as e:
            logger.error("Erreur lors de la mise à jour périodique: %s", e)
        await asyncio.sleep(UPDATE_INTERVAL)


def _prepare_query_vector(query_vector: np.ndarray) -> np.ndarray:
    """Prépare le vecteur de requête avec la bonne dimension.

    Args:
        query_vector: Le vecteur de requête à préparer

    Returns:
        np.ndarray: Vecteur de requête redimensionné si nécessaire

    Raises:
        FAISSLoadError: Si l'index FAISS n'est pas initialisé
    """
    if _state.index is None:
        raise FAISSLoadError("Index FAISS non initialisé")

    # Log détaillé des dimensions pour faciliter le débogage
    logger.debug(
        "Préparation du vecteur de requête - Dimensions: vecteur=%s, index=%d",
        query_vector.shape,
        _state.index.d,
    )

    if query_vector.shape[1] != _state.index.d:
        logger.warning(
            "Différence de dimensions détectée: vecteur=%d, index=%d. Ajustement automatique...",
            query_vector.shape[1],
            _state.index.d,
        )

        new_vector = np.zeros((1, _state.index.d), dtype="float32")
        min_dim = min(query_vector.shape[1], _state.index.d)
        new_vector[0, :min_dim] = query_vector[0, :min_dim]
        return new_vector

    return query_vector


def _search_in_index(query_vector: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Effectue la recherche dans l'index FAISS."""
    if _state.index is None:
        raise FAISSLoadError("Index FAISS non disponible")
    try:
        return _state.index.search(query_vector, k)
    except RuntimeError as e:
        raise FAISSServiceError(f"Erreur lors de la recherche FAISS: {str(e)}") from e


def _process_search_results(
    distances: np.ndarray, indices: np.ndarray
) -> list[dict[str, Any]]:
    """Traite les résultats de la recherche FAISS."""
    results = []

    for i, idx in enumerate(indices[0]):
        if idx < 0 or str(idx) not in _state.document_store:
            continue

        doc = _state.document_store[str(idx)]
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


def retrieve_similar_documents(query: str, k: int = 5) -> list[dict[str, Any]]:
    """Recherche les documents les plus similaires à la requête.

    Args:
        query: La requête texte
        k: Nombre de résultats à retourner

    Returns:
        list[dict[str, Any]]: Liste des documents similaires avec leurs métadonnées
    """
    if _state.index is None:
        try:
            index, doc_store = load_index()
            _state.index = index
            _state.document_store = doc_store
            if _state.index is None:
                logger.warning("Index FAISS non disponible")
                return []
        except (FAISSServiceError, OSError, RuntimeError) as e:
            # En environnement de test, on peut avoir des erreurs si le fichier n'existe pas
            logger.warning("Erreur lors du chargement de l'index FAISS: %s", e)
            return []
    try:
        query_vector = generate_query_vector(query)
        prepared_vector = _prepare_query_vector(query_vector)
        distances, indices = _search_in_index(prepared_vector, k)
        return _process_search_results(distances, indices)
    except (ValueError, FAISSServiceError, RuntimeError) as e:
        logger.warning("Erreur lors de la recherche: %s", e)
        return []
