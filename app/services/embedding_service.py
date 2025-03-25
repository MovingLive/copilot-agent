"""Service de gestion des embeddings utilisant SentenceTransformers.

Fournit les fonctionnalités de génération de vecteurs d'embeddings pour la recherche similaire.
"""

import logging

import numpy as np
import torch
from fastapi import HTTPException
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# Constantes
MODEL_NAME = "all-MiniLM-L6-v2"
EXPECTED_DIMENSION = 384
HTTP_500_ERROR = "Erreur interne du service d'embeddings"

# Variables globales du module
_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Retourne le modèle d'embedding (lazy loading).

    Returns:
        SentenceTransformer: Instance du modèle d'embedding

    Raises:
        HTTPException: Si le modèle ne peut pas être chargé
    """
    global _model
    if _model is None:
        try:
            logger.info("Chargement du modèle d'embedding '%s'...", MODEL_NAME)
            _model = SentenceTransformer(MODEL_NAME)
            logger.info(
                "Modèle chargé avec succès, dimension=%d",
                _model.get_sentence_embedding_dimension(),
            )
        except Exception as e:
            logger.error("Erreur lors du chargement du modèle: %s", e)
            raise HTTPException(
                status_code=500,
                detail=f"{HTTP_500_ERROR}: impossible de charger le modèle",
            ) from e
    return _model


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
        model = get_embedding_model()

        with torch.no_grad():
            embedding = model.encode(
                text, convert_to_tensor=True, normalize_embeddings=True
            )

        embedding = embedding.cpu().numpy()
        embedding = normalize_vector(embedding)

        if embedding.shape[1] != EXPECTED_DIMENSION:
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
        np.ndarray: Vecteur de requête normalisé

    Raises:
        ValueError: Si la requête est invalide
        HTTPException: En cas d'erreur de génération
    """
    try:
        validate_input(query)
        model = get_embedding_model()

        with torch.no_grad():
            vector = (
                model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
                .cpu()
                .numpy()
            )

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
