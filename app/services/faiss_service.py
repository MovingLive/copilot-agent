"""Service de gestion des opérations FAISS.

Fournit les fonctionnalités de recherche vectorielle avec FAISS.
"""

import asyncio
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3
import faiss
import numpy as np
from botocore.exceptions import BotoCoreError, ClientError

from app.core.config import settings
from app.services.embedding_service import generate_query_vector
from app.services.vector_cache import get_cache_instance
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
    base_dir = (
        settings.LOCAL_OUTPUT_DIR if is_local_environment() else tempfile.mkdtemp()
    )
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


def create_optimized_index(dimension: int, vector_count: int) -> faiss.Index:
    """Crée un index FAISS optimisé basé sur la taille des données.

    Args:
        dimension: Dimension des vecteurs
        vector_count: Nombre approximatif de vecteurs attendus

    Returns:
        faiss.Index: Index FAISS optimisé
    """
    # Déterminer le meilleur type d'index en fonction de la taille des données
    if vector_count < 10000:
        # Pour petits ensembles: recherche exacte
        logger.info(
            f"Création d'un index exact (IndexFlatL2) pour {vector_count} vecteurs"
        )
        return faiss.IndexFlatL2(dimension)
    elif vector_count < 100000:
        # Pour ensembles moyens: IVF avec clusters
        logger.info(f"Création d'un index IVF pour {vector_count} vecteurs")
        n_clusters = min(int(4 * np.sqrt(vector_count)), 8192)  # Règle empirique
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, n_clusters)
        index.nprobe = 8  # Valeur par défaut, peut être ajustée lors de la recherche
        return index
    else:
        # Pour grands ensembles: HNSW pour rapidité et précision
        logger.info(f"Création d'un index HNSW pour {vector_count} vecteurs")
        index = faiss.IndexHNSWFlat(dimension, 32)  # 32 voisins par nœud
        index.hnsw.efConstruction = (
            40  # Plus élevé = plus précis mais plus lent à construire
        )
        index.hnsw.efSearch = 16  # Plus élevé = plus précis mais plus lent à rechercher
        return index


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


def configure_search_parameters(
    k: int, precision_priority: bool = False
) -> dict[str, Any]:
    """Configure les paramètres de recherche FAISS optimaux.

    Args:
        k: Nombre de résultats à retourner
        precision_priority: Si vrai, privilégie la précision à la vitesse

    Returns:
        Dict[str, Any]: Paramètres optimisés pour la recherche
    """
    params = {"k": k}

    # Adaptation des paramètres selon le type d'index
    if _state.index is None:
        return params

    # Pour les index IVF, configure nprobe (nombre de cellules à explorer)
    if isinstance(_state.index, faiss.IndexIVFFlat):
        # Augmente nprobe pour améliorer le rappel, au détriment de la vitesse
        nprobe = 8  # Valeur par défaut
        if precision_priority:
            # Si on veut plus de précision, on augmente nprobe
            nprobe = min(32, max(16, k * 2))  # Au moins 16, au plus 32

        _state.index.nprobe = nprobe
        params["nprobe"] = nprobe

    # Pour HNSW, configure efSearch
    elif hasattr(_state.index, "hnsw"):
        ef_search = 16  # Valeur par défaut
        if precision_priority:
            ef_search = min(80, max(40, k * 4))  # Au moins 40, au plus 80

        _state.index.hnsw.efSearch = ef_search
        params["ef_search"] = ef_search

    logger.debug("Paramètres de recherche configurés: %s", params)
    return params


def _search_in_index(
    query_vector: np.ndarray, k: int, precision_priority: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Effectue la recherche dans l'index FAISS avec paramètres optimisés."""
    if _state.index is None:
        raise FAISSLoadError("Index FAISS non disponible")

    # Configure les paramètres de recherche optimaux
    search_params = configure_search_parameters(k, precision_priority)
    logger.info("🔍 Recherche avec paramètres: %s", search_params)

    try:
        distances, indices = _state.index.search(query_vector, k)
        
        # Log détaillé des résultats bruts
        valid_indices = [idx for idx in indices[0] if idx >= 0]
        logger.info(
            "🔍 Résultats bruts: %d résultats valides sur %d demandés", 
            len(valid_indices), k
        )
        
        if len(valid_indices) > 0:
            min_dist = np.min(distances[0][distances[0] > 0])
            max_dist = np.max(distances[0])
            logger.info(
                "📏 Distances: min=%.4f, max=%.4f, moyenne=%.4f", 
                min_dist, max_dist, np.mean(distances[0][distances[0] > 0])
            )
            
            # Log des 3 premiers indices et distances pour débogage
            for i in range(min(3, len(valid_indices))):
                idx = indices[0][i]
                if idx >= 0:
                    logger.info(
                        "🏆 Top %d: ID=%d, distance=%.4f", 
                        i+1, idx, distances[0][i]
                    )
        
        return distances, indices
    except RuntimeError as e:
        logger.error("❌ Erreur lors de la recherche FAISS: %s", e)
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


def retrieve_similar_documents(
    query: str, k: int = 5, precision_priority: bool = False, use_cache: bool = True
) -> list[dict[str, Any]]:
    """Recherche les documents les plus similaires à la requête.

    Args:
        query: La requête texte
        k: Nombre de résultats à retourner
        precision_priority: Si True, privilégie la précision à la vitesse
        use_cache: Si True, utilise le cache pour les résultats de recherche

    Returns:
        list[dict[str, Any]]: Liste des documents similaires avec leurs métadonnées
    """
    # Vérifier d'abord dans le cache si cette requête a déjà été traitée
    if use_cache:
        cache = get_cache_instance()
        cached_results = cache.get_search_results(query)
        if cached_results is not None:
            logger.info(
                "Résultats trouvés dans le cache pour la requête: %s", query[:30]
            )
            return cached_results

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
        distances, indices = _search_in_index(prepared_vector, k, precision_priority)
        results = _process_search_results(distances, indices)

        # Stockage des résultats dans le cache pour les requêtes futures
        if use_cache and results:
            cache = get_cache_instance()
            cache.store_search_results(query, results)

        return results
    except (ValueError, FAISSServiceError, RuntimeError) as e:
        logger.warning("Erreur lors de la recherche: %s", e)
        return []


def save_faiss_index(
    index: faiss.Index, metadata_mapping: dict, directory: str
) -> None:
    """Sauvegarde l'index FAISS et le mapping des IDs dans le répertoire spécifié.

    Args:
        index: Index FAISS à sauvegarder
        metadata_mapping: Mapping des IDs vers les métadonnées
        directory: Répertoire de destination

    Raises:
        PermissionError: Si les permissions sont insuffisantes pour créer/écrire les fichiers
        OSError: Pour les autres erreurs d'E/S
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Répertoire créé: {directory}")

        # Convertir les clés en str pour la sérialisation JSON
        str_mapping = {str(k): v for k, v in metadata_mapping.items()}
        index_file_path = os.path.join(directory, settings.FAISS_INDEX_FILE)
        mapping_file_path = os.path.join(directory, settings.FAISS_METADATA_FILE)

        # Sauvegarder l'index et le mapping
        faiss.write_index(index, index_file_path)
        logger.info(f"Index FAISS sauvegardé: {index_file_path}")

        with open(mapping_file_path, "w", encoding="utf-8") as f:
            json.dump(str_mapping, f, ensure_ascii=False, indent=2)

    except (PermissionError, OSError) as e:
        logger.error(f"Erreur lors de la sauvegarde de l'index : {e}")
        raise
