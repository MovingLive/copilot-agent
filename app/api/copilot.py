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
from app.utils.translation_utils import (
    detect_language,
    needs_translation,
    translate_text,
)

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

    # Détection de la langue de la question
    query_lang = detect_language(query)
    logger.debug("Langue détectée de la question: %s", query_lang)

    # Traduction de la question si nécessaire pour correspondre à la langue des documents
    translated_query = query
    if needs_translation(query, settings.FAISS_LANG):
        logger.info(
            "Traduction de la question de '%s' vers '%s'",
            query_lang,
            settings.FAISS_LANG,
        )
        translated_query = translate_text(
            query, src_lang=query_lang, tgt_lang=settings.FAISS_LANG
        )
        logger.debug("Question traduite: %s", translated_query)

    # Traduction du contexte additionnel si présent et nécessaire
    translated_additional_context = additional_context
    if additional_context and needs_translation(
        additional_context, settings.FAISS_LANG
    ):
        translated_additional_context = translate_text(
            additional_context, tgt_lang=settings.FAISS_LANG
        )

    # Recherche de documents similaires avec la question traduite
    search_query = (
        translated_query + " " + translated_additional_context
        if translated_additional_context
        else translated_query
    )
    docs = retrieve_similar_documents(search_query, k=5)

    # Vérification que nous avons des documents valides
    if not docs:
        logger.warning("Aucun document similaire trouvé pour la requête")

    # Récupération des informations de l'utilisateur
    user_login = await get_github_user(auth_token)

    # Préparation des messages pour Copilot (avec la question originale, non traduite)
    formatted_messages = format_copilot_messages(query, docs, user_login)
    data["messages"] = formatted_messages

    # Génération de la réponse en streaming
    return StreamingResponse(
        generate_streaming_response(data, auth_token), media_type="application/json"
    )


@router.post("/")
async def query_copilot(request: Request) -> StreamingResponse:
    """Endpoint principal pour l'API Copilot."""
    return await handle_copilot_query(request)
