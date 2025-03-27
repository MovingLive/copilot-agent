"""Service de cache pour les embeddings vectoriels et les résultats de recherche.

Ce module fournit un système de cache pour éviter de recalculer les embeddings
et de refaire les recherches pour des requêtes répétitives.
"""

import hashlib
import logging
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class VectorCache:
    """Cache pour éviter de recalculer les embeddings et les résultats de recherche."""

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """Initialise un nouveau cache d'embeddings et de résultats.

        Args:
            max_size: Taille maximale du cache (nombre d'entrées)
            ttl_seconds: Durée de vie des entrées du cache en secondes
        """
        self.embedding_cache = OrderedDict()  # text_hash -> vector
        self.results_cache = OrderedDict()  # query_hash -> results
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.last_access = {}  # Pour implémentation LRU et TTL
        self.hit_count = 0
        self.miss_count = 0

    def get_embedding(self, text: str) -> np.ndarray | None:
        """Récupère un embedding du cache s'il existe.

        Args:
            text: Texte pour lequel on cherche l'embedding

        Returns:
            np.ndarray ou None: Le vecteur d'embedding ou None si absent du cache
        """
        text_hash = self._hash_text(text)

        if text_hash in self.embedding_cache:
            self.hit_count += 1
            self._update_access(text_hash)
            return self.embedding_cache[text_hash]

        self.miss_count += 1
        return None

    def store_embedding(self, text: str, vector: np.ndarray) -> None:
        """Stocke un embedding dans le cache.

        Args:
            text: Texte source de l'embedding
            vector: Vecteur d'embedding à stocker
        """
        text_hash = self._hash_text(text)
        self._ensure_cache_size(self.embedding_cache)
        self.embedding_cache[text_hash] = vector
        self._update_access(text_hash)

    def get_search_results(self, query: str) -> list[dict[str, Any]] | None:
        """Récupère des résultats de recherche du cache.

        Args:
            query: Requête pour laquelle on cherche les résultats

        Returns:
            Liste de résultats ou None si absents du cache ou expirés
        """
        query_hash = self._hash_text(query)

        if query_hash in self.results_cache:
            # Vérifier si les résultats sont encore frais
            timestamp = self.last_access.get(query_hash, 0)
            if time.time() - timestamp < self.ttl_seconds:
                self.hit_count += 1
                self._update_access(query_hash)
                return self.results_cache[query_hash]
            else:
                # Résultats expirés, les supprimer
                del self.results_cache[query_hash]
                if query_hash in self.last_access:
                    del self.last_access[query_hash]

        self.miss_count += 1
        return None

    def store_search_results(self, query: str, results: list[dict[str, Any]]) -> None:
        """Stocke des résultats de recherche dans le cache.

        Args:
            query: Requête source des résultats
            results: Résultats de recherche à stocker
        """
        query_hash = self._hash_text(query)
        self._ensure_cache_size(self.results_cache)
        self.results_cache[query_hash] = results
        self._update_access(query_hash)

    def get_cache_stats(self) -> dict[str, Any]:
        """Retourne des statistiques sur l'utilisation du cache.

        Returns:
            Dict contenant les statistiques du cache
        """
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0

        stats = {
            "embedding_cache_size": len(self.embedding_cache),
            "results_cache_size": len(self.results_cache),
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": hit_rate,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        return stats

    def clear(self) -> None:
        """Vide le cache."""
        self.embedding_cache.clear()
        self.results_cache.clear()
        self.last_access.clear()
        logger.info("Cache vidé")

    def _hash_text(self, text: str) -> str:
        """Génère un hash pour un texte.

        Args:
            text: Texte à hasher

        Returns:
            str: Hash du texte
        """
        # Utilisation de SHA-256 au lieu de MD5 pour des raisons de sécurité
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _update_access(self, key: str) -> None:
        """Met à jour l'horodatage du dernier accès à une entrée.

        Args:
            key: Clé de l'entrée accédée
        """
        self.last_access[key] = time.time()

    def _ensure_cache_size(self, cache: OrderedDict) -> None:
        """Assure que le cache ne dépasse pas la taille maximale (LRU).

        Args:
            cache: Le cache à vérifier (embedding_cache ou results_cache)
        """
        while len(cache) >= self.max_size:
            # Trouver l'élément le moins récemment utilisé
            oldest_key = None
            oldest_time = float("inf")

            for key, access_time in self.last_access.items():
                if key in cache and access_time < oldest_time:
                    oldest_key = key
                    oldest_time = access_time

            if oldest_key:
                if oldest_key in cache:
                    del cache[oldest_key]
                if oldest_key in self.last_access:
                    del self.last_access[oldest_key]
                logger.debug("Élément supprimé du cache (LRU): %s", oldest_key[:8])
            else:
                # Si on ne trouve pas de candidat, supprimer le premier élément
                try:
                    key, _ = cache.popitem(last=False)
                    if key in self.last_access:
                        del self.last_access[key]
                except KeyError:
                    pass  # Cache déjà vide


# Singleton pour le cache
_vector_cache = VectorCache(max_size=2000, ttl_seconds=7200)


def get_cache_instance() -> VectorCache:
    """Retourne l'instance unique du cache.

    Returns:
        VectorCache: L'instance du cache
    """
    return _vector_cache
