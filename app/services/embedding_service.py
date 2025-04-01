"""Service de gestion des embeddings utilisant SentenceTransformers.

Fournit les fonctionnalités de génération de vecteurs d'embeddings pour la recherche similaire.
"""

import logging
from typing import Optional

import numpy as np
import torch
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

from app.services.vector_cache import get_cache_instance

logger = logging.getLogger(__name__)

# Constantes
MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384
HTTP_500_ERROR = "Erreur interne du service d'embeddings"


class EmbeddingService:
    """Service pour gérer les embeddings avec SentenceTransformer."""

    _instance: Optional["EmbeddingService"] = None
    _model: SentenceTransformer | None = None

    def __new__(cls) -> "EmbeddingService":
        """Implémentation du pattern singleton."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> "EmbeddingService":
        """Retourne l'instance singleton du service.

        Returns:
            EmbeddingService: Instance unique du service
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        """Getter pour le modèle d'embedding (lazy loading).

        Returns:
            SentenceTransformer: Instance du modèle

        Raises:
            HTTPException: Si le modèle ne peut pas être chargé
        """
        if self._model is None:
            try:
                logger.info("Chargement du modèle d'embedding '%s'...", MODEL_NAME)
                self._model = SentenceTransformer(MODEL_NAME)
                logger.info(
                    "Modèle chargé avec succès, dimension=%d",
                    self._model.get_sentence_embedding_dimension(),
                )
            except Exception as e:
                logger.error("Erreur lors du chargement du modèle: %s", e)
                raise HTTPException(
                    status_code=500,
                    detail=f"{HTTP_500_ERROR}: impossible de charger le modèle",
                ) from e
        return self._model

    @model.setter
    def model(self, model: SentenceTransformer) -> None:
        """Setter pour le modèle d'embedding (utilisé pour les tests).

        Args:
            model: Instance du modèle à utiliser
        """
        self._model = model


def validate_input(text: str) -> None:
    """Valide le texte d'entrée.

    Args:
        text: Texte à valider

    Raises:
        ValueError: Si le texte est invalide
    """
    if text is None:
        raise ValueError("Le texte ne peut pas être None")
    if not text.strip():
        raise ValueError("Le texte ne peut pas être vide")


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normalise un vecteur d'embedding.

    Args:
        vector: Vecteur à normaliser

    Returns:
        np.ndarray: Vecteur normalisé
    """
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    return vector


def embed_text(text: str) -> list[float]:
    """Génère un embedding pour un texte donné.

    Args:
        text: Texte à transformer en embedding

    Returns:
        list[float]: Vecteur d'embedding normalisé

    Raises:
        HTTPException: En cas d'erreur lors de la génération
        ValueError: Si le texte est invalide
    """
    try:
        validate_input(text)

        # Vérifier si l'embedding est déjà dans le cache
        cache = get_cache_instance()
        cached_embedding = cache.get_embedding(text)

        if cached_embedding is not None:
            logger.debug("Embedding trouvé dans le cache")
            return cached_embedding.tolist()

        # Si non trouvé en cache, calculer l'embedding
        model = EmbeddingService.get_instance().model
        with torch.no_grad():
            embedding = model.encode(
                text, convert_to_tensor=True, normalize_embeddings=True
            )
        embedding = embedding.cpu().numpy()
        embedding = normalize_vector(embedding)

        # Vérifier la dimension selon que c'est un vecteur 1D ou 2D
        if embedding.ndim == 1:
            if embedding.shape[0] != EXPECTED_DIMENSION:
                raise ValueError(
                    f"Dimension incorrecte: {embedding.shape[0]}, "
                    f"attendu: {EXPECTED_DIMENSION}"
                )
        elif embedding.shape[1] != EXPECTED_DIMENSION:
            raise ValueError(
                f"Dimension incorrecte: {embedding.shape[1]}, "
                f"attendu: {EXPECTED_DIMENSION}"
            )

        # Stocker l'embedding dans le cache
        cache.store_embedding(text, embedding)

        logger.info(
            "Embedding généré avec succès: min=%f, max=%f, norme=%f",
            np.min(embedding),
            np.max(embedding),
            np.linalg.norm(embedding),
        )

        return embedding.tolist()
    except ValueError as ve:
        logger.error("Erreur de validation: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        logger.error("Erreur lors de la génération de l'embedding: %s", e)
        raise HTTPException(
            status_code=500, detail=f"{HTTP_500_ERROR}: {str(e)}"
        ) from e


def generate_query_vector(query: str) -> np.ndarray:
    """Génère un vecteur de requête pour la recherche.

    Args:
        query: Texte de la requête

    Returns:
        np.ndarray: Vecteur de requête normalisé et correctement dimensionné (2D)

    Raises:
        ValueError: Si la requête est invalide
        HTTPException: En cas d'erreur de génération
    """
    try:
        validate_input(query)
        logger.info("Génération du vecteur de requête pour: '%s'", query[:50] + "..." if len(query) > 50 else query)

        # Vérifier si le vecteur est déjà dans le cache
        cache = get_cache_instance()
        cached_vector = cache.get_embedding(query)

        if cached_vector is not None:
            logger.info("✅ Vecteur de requête trouvé dans le cache")

            # S'assurer que le format est correct (2D)
            if cached_vector.ndim == 1:
                cached_vector = cached_vector.reshape(1, -1)

            # Log des statistiques du vecteur
            norm = np.linalg.norm(cached_vector)
            mean = np.mean(cached_vector)
            std = np.std(cached_vector)
            logger.info(
                "📊 Statistiques du vecteur (cache): norme=%.4f, moyenne=%.4f, écart-type=%.4f",
                norm, mean, std
            )
            return cached_vector

        # Si non trouvé en cache, calculer le vecteur
        model = EmbeddingService.get_instance().model
        with torch.no_grad():
            # Force la génération d'embeddings en mode batch (1 élément) pour assurer un tenseur 2D
            vector = (
                model.encode([query], convert_to_tensor=True, normalize_embeddings=True)
                .cpu()
                .numpy()
            )

        # Vérification de la dimension attendue
        if vector.shape[1] != EXPECTED_DIMENSION:
            raise ValueError(
                f"Dimension incorrecte: {vector.shape[1]}, "
                f"attendu: {EXPECTED_DIMENSION}"
            )

        vector = normalize_vector(vector)

        # Stocker le vecteur dans le cache
        cache.store_embedding(query, vector)

        # Log détaillé sur le vecteur généré
        norm = np.linalg.norm(vector)
        mean = np.mean(vector)
        std = np.std(vector)
        min_val = np.min(vector)
        max_val = np.max(vector)
        non_zeros = np.count_nonzero(vector)
        logger.info(
            "📊 Statistiques du vecteur (généré): dimension=%s, norme=%.4f, moyenne=%.4f, écart-type=%.4f",
            vector.shape, norm, mean, std
        )
        logger.info(
            "📈 Détails supplémentaires: min=%.4f, max=%.4f, valeurs non-nulles=%d/%d",
            min_val, max_val, non_zeros, vector.size
        )

        return vector
    except ValueError as ve:
        logger.error("❌ Erreur de validation: %s", ve)
        raise
    except Exception as e:
        logger.error("❌ Erreur lors de la génération du vecteur: %s", e)
        raise HTTPException(
            status_code=500, detail=f"{HTTP_500_ERROR}: {str(e)}"
        ) from e
