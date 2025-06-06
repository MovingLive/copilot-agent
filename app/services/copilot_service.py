"""Service de gestion des interactions avec l'API Copilot."""

import logging
from http import HTTPStatus
from typing import Any

import httpx
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)


def format_copilot_messages(query: str, context: list[dict[str, Any]]) -> list[dict]:
    """Formate les messages pour l'API Copilot.

    Args:
        query: La requête de l'utilisateur
        context: Liste des documents contextuels

    Returns:
        Liste de messages formatés pour l'API Copilot
    """
    # Construction simple du contexte
    context_text = "Aucune information contextuelle disponible."

    if context:
        context_text = ""
        for i, doc in enumerate(context, 1):
            metadata = doc.get("metadata", {})
            source = metadata.get("source", "Source inconnue")

            content = doc.get("content", "").strip()
            if content:
                context_text += f"[{i}] {source}: {content[:300]}...\n\n"

    return [
        {
            "role": "system",
            "content": "Tu es un assistant spécialisé dans les outils DevSecOps.",
        },
        {
            "role": "system",
            "content": f"Utilise ces informations factuelles pour répondre:\n\n{context_text}",
        },
        {
            "role": "system",
            "content": "Structure ta réponse avec: 1) Une réponse directe, 2) Des explications détaillées, 3) Un exemple pratique si pertinent.",
        },
        {"role": "user", "content": query},
    ]


async def call_copilot_api(messages: list[dict], auth_token: str) -> str:
    """Appelle l'API Copilot avec les messages formatés."""
    headers = {
        "authorization": f"Bearer {auth_token}",
        "content-type": "application/json",
    }
    payload = {"messages": messages}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                settings.COPILOT_API_URL, json=payload, headers=headers, timeout=30.0
            )
            if not response.is_success:
                logger.error(
                    "Échec de l'appel API Copilot (%d): %s",
                    response.status_code,
                    response.text,
                )
            response.raise_for_status()
            data = response.json()  # Utilisation synchrone pour les tests
            if not data.get("choices") or "message" not in data["choices"][0]:
                # On laisse propager l'exception ValueError pour le test
                logger.error("Format de réponse inattendu de l'API Copilot")
                raise ValueError("Format de réponse inattendu")
            return data["choices"][0]["message"]["content"]
    except httpx.HTTPError as http_err:
        handle_copilot_api_error(http_err)
    except ValueError:
        # On propage l'erreur ValueError au lieu de la convertir en HTTPException
        raise
    except Exception as e:
        logger.error("Erreur lors de l'appel à l'API Copilot: %s", e)
        raise HTTPException(
            status_code=500, detail="Erreur lors de l'appel au service Copilot"
        ) from e


def handle_copilot_api_error(error: httpx.HTTPError) -> None:
    """Gère les erreurs spécifiques de l'API Copilot."""
    if isinstance(error, httpx.NetworkError | httpx.TimeoutException):
        raise HTTPException(
            status_code=500,
            detail=f"Erreur de connexion au service Copilot: {str(error)}",
        )
    if not hasattr(error, "response"):
        raise HTTPException(
            status_code=500,
            detail=f"Erreur inattendue du service Copilot: {str(error)}",
        )

    if error.response.status_code == HTTPStatus.BAD_REQUEST:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST,
            detail="Format de requête incorrect pour l'API Copilot",
        )
    elif error.response.status_code == HTTPStatus.UNAUTHORIZED:
        raise HTTPException(
            status_code=HTTPStatus.UNAUTHORIZED,
            detail="Token d'authentification Copilot invalide ou expiré",
        )
    else:
        raise HTTPException(
            status_code=error.response.status_code,
            detail=f"Erreur Copilot: {error.response.text}",
        )


async def handle_streaming_error(response: httpx.Response) -> None:
    """Gère les erreurs de streaming depuis l'API Copilot."""
    error = httpx.HTTPStatusError(
        "Streaming error",
        request=httpx.Request("POST", settings.COPILOT_API_URL),
        response=response,
    )
    handle_copilot_api_error(error)


async def generate_streaming_response(request_data: dict, auth_token: str):
    """Génère une réponse en streaming depuis l'API Copilot."""
    headers = {
        "authorization": f"Bearer {auth_token}",
        "content-type": "application/json",
    }
    payload = {"messages": request_data["messages"], "stream": True}

    try:
        async with (
            httpx.AsyncClient() as client,
            client.stream(
                "POST",
                settings.COPILOT_API_URL,
                headers=headers,
                json=payload,
                timeout=None,
            ) as response,
        ):
            if not response.is_success:
                await handle_streaming_error(response)
            async for chunk in response.aiter_bytes():
                yield chunk
    except httpx.HTTPError as http_err:
        handle_copilot_api_error(http_err)
    except HTTPException:
        # Propager directement les HTTPException déjà formatées
        raise
    except Exception as e:
        logger.error("Erreur lors du streaming depuis l'API Copilot: %s", e)
        raise HTTPException(
            status_code=500, detail="Erreur lors du streaming depuis le service Copilot"
        ) from e
