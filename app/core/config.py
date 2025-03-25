"""Configuration centralisée de l'application.

Utilise pydantic pour la validation des configurations.
"""

import logging
import os

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict

# Chargement des variables d'environnement
load_dotenv()


class Settings(BaseModel):
    """Configuration centralisée de l'application."""

    model_config = ConfigDict(from_attributes=True)

    # Configuration de l'application
    APP_TITLE: str = "Copilot API LLM"
    APP_DESCRIPTION: str = "API pour interagir avec GitHub Copilot LLM"
    APP_VERSION: str = "1.0.0"
    LOG_LEVEL: int = getattr(logging, os.getenv("LOG_LEVEL", "INFO"))

    # Configuration CORS
    CORS_ORIGINS: list[str] = ["*"]
    CORS_METHODS: list[str] = ["*"]
    CORS_HEADERS: list[str] = ["*"]

    # Environnement
    ENV: str = os.getenv("ENV", "local")

    # Configuration FAISS
    FAISS_INDEX_FILE: str = os.getenv("FAISS_INDEX_FILE", "index.faiss")
    FAISS_METADATA_FILE: str = os.getenv("FAISS_METADATA_FILE", "metadata.json")
    FAISS_PERSIST_DIR: str = "faiss_index"

    # Configuration temporaire
    TEMP_FAISS_DIR: str = os.getenv(
        "TEMP_FAISS_DIR",
        os.path.join(
            os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            ),
            "temp",
            "faiss_index",
        ),
    )

    # Configuration GitHub
    REPO_URL: str = os.getenv(
        "REPO_URL", "https://github.com/votre_utilisateur/votre_repo.git"
    )
    REPO_DIR: str = os.getenv("REPO_DIR", "documentation_repo")

    # Configuration AWS
    S3_BUCKET_NAME: str = os.getenv("S3_BUCKET_NAME", "mon-bucket-faiss")
    S3_BUCKET_PREFIX: str = "faiss_index"
    AWS_REGION: str = os.getenv("AWS_REGION", "ca-central-1")

    # Configuration API Copilot
    COPILOT_API_URL: str = os.getenv(
        "COPILOT_API_URL", "https://api.githubcopilot.com/chat/completions"
    )
    COPILOT_TOKEN: str = os.getenv("COPILOT_TOKEN", "")

    # Configuration de segmentation
    SEGMENT_MAX_LENGTH: int = int(os.getenv("SEGMENT_MAX_LENGTH", "1000"))
    MIN_SEGMENT_LENGTH: int = 10
    MAX_SUMMARY_LENGTH: int = 200

    # Chemins de fichiers
    LOCAL_OUTPUT_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "output",
    )
    CHROMA_PERSIST_DIR: str = "chroma_db"
    COLLECTION_NAME: str = "documentation"

    # Constantes pour la recherche de similarité
    MAX_DISTANCE_THRESHOLD: float = 100.0
    TOP_K_RESULTS: int = 5

    # Configuration de logging
    LOG_FORMAT: str = "%(asctime)s [%(levelname)s] %(message)s"


# Instance singleton des configurations
settings = Settings()
