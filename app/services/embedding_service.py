"""Service de gestion des embeddings utilisant SentenceTransformers.

Fournit les fonctionnalités de génération de vecteurs d'embeddings pour la recherche similaire.
"""

import logging
from typing import Optional

import numpy as np
import torch
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

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
        logger.debug(
            "Vecteur de requête généré: dimension=%s, norme=%f",
            vector.shape,
            np.linalg.norm(vector),
        )
        return vector

    except ValueError as ve:
        logger.error("Erreur de validation: %s", ve)
        raise
    except Exception as e:
        logger.error("Erreur lors de la génération du vecteur: %s", e)
        raise HTTPException(
            status_code=500, detail=f"{HTTP_500_ERROR}: {str(e)}"
        ) from e
