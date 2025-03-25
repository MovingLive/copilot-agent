"""Routeur FastAPI pour les endpoints Copilot."""

import logging

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.config import settings
from app.services.copilot_service import (
    format_copilot_messages,
    generate_streaming_response,
    get_github_user,
)
from app.services.faiss_service import retrieve_similar_documents

router = APIRouter()
logger = logging.getLogger(__name__)


async def handle_copilot_query(request: Request) -> StreamingResponse:
    """Gère les requêtes à l'API Copilot."""
    # Vérification du token d'authentification
    auth_token = request.headers.get("x-github-token")
    if not auth_token:
        raise HTTPException(status_code=401, detail="Token d'authentification manquant")

    # Récupération et validation des données de la requête
    data = await request.json()
    messages = data.get("messages", [])
    if not messages:
        raise HTTPException(
            status_code=400, detail="Messages manquants dans la requête"
        )

    # Extraction de la dernière question
    query = messages[-1].get("content", "")
    if not query:
        raise HTTPException(status_code=400, detail="Message vide")

    # Récupération du contexte additionnel
    additional_context = data.get("copilot_references", "")

    # Recherche de documents similaires
    docs = retrieve_similar_documents(query + " " + additional_context, k=5)
    context_sections = [
        doc["content"]
        for doc in docs
        if doc.get("content") and len(doc["content"]) > settings.MIN_SEGMENT_LENGTH
    ]

    # Formatage du contexte pour le LLM
    formatted_context = "\n\n".join(
        [
            "CONTEXTE PERTINENT:",
            *context_sections,
            "\nINSTRUCTIONS:",
            "- Utilise le contexte ci-dessus pour répondre à la question.",
            "- Si le contexte contient une réponse directe, base ta réponse dessus.",
            "- Structure ta réponse de façon claire et précise.",
        ]
    )

    # Récupération des informations de l'utilisateur
    user_login = await get_github_user(auth_token)

    # Préparation des messages pour Copilot
    formatted_messages = format_copilot_messages(query, formatted_context, user_login)
    data["messages"] = formatted_messages

    # Génération de la réponse en streaming
    return StreamingResponse(
        generate_streaming_response(data, auth_token), media_type="application/json"
    )


@router.post("/")
async def query_copilot(request: Request) -> StreamingResponse:
    """Endpoint principal pour l'API Copilot."""
    return await handle_copilot_query(request)
