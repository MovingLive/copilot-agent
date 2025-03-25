"""Point d'entrée principal de l'application FastAPI.

Orchestre les différents composants de l'application.
"""

import logging
import threading
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import copilot, health
from app.core.config import settings
from app.services import embedding_service, faiss_service

# Configuration du logging
logging.basicConfig(
    level=settings.LOG_LEVEL, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("copilot_api")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Gère le cycle de vie de l'application FastAPI.

    Initialise les services au démarrage et nettoie les ressources à l'arrêt.
    """
    try:
        # Initialisation du modèle d'embedding
        logger.info("Initialisation du modèle d'embedding...")
        embedding_service.get_embedding_model()

        # Initialisation de FAISS
        logger.info("Initialisation du service FAISS...")
        faiss_service.load_faiss_index()

        # Démarrage du service de mise à jour périodique
        logger.info("Démarrage du service de mise à jour périodique...")
        update_thread = threading.Thread(
            target=faiss_service.update_faiss_index_periodically,
            daemon=True,
            name="faiss_updater",
        )
        update_thread.start()

    except Exception as e:
        logger.error("Erreur lors de l'initialisation des services: %s", e)
        raise  # Propager l'erreur pour empêcher le démarrage si critique

    yield

    logger.info("Arrêt de l'application")


# Création de l'instance FastAPI avec le gestionnaire de cycle de vie
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=settings.CORS_METHODS,
    allow_headers=settings.CORS_HEADERS,
)

# Montage des routeurs
app.include_router(health.router, prefix="/api", tags=["health"])
app.include_router(copilot.router, prefix="/api", tags=["copilot"])
