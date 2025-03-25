"""Routeur FastAPI pour les endpoints de santé."""

from fastapi import APIRouter

router = APIRouter()


@router.get("/")
async def root() -> dict:
    """Point d'entrée principal de l'API."""
    return {"message": "Bienvenue dans l'API Copilot LLM!"}


@router.get("/health")
async def health_check() -> dict:
    """Endpoint de vérification de santé."""
    return {"status": "healthy"}
