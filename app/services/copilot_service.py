"""Service de gestion des interactions avec l'API Copilot."""

import logging

import httpx
from fastapi import HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)


async def get_github_user(auth_token: str) -> str:
    """Récupère les informations de l'utilisateur GitHub.

    Args:
        auth_token: Token d'authentification GitHub

    Returns:
        str: Login de l'utilisateur

    Raises:
        HTTPException: En cas d'erreur d'authentification
    """
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            response.raise_for_status()
            user_data = await response.json()
            return user_data.get("login")
        except Exception as e:
            logger.error("Erreur lors de l'authentification GitHub: %s", e)
            raise HTTPException(status_code=401, detail="Token GitHub invalide") from e


def format_copilot_messages(query: str, context: str, user_login: str) -> list[dict]:
    """Formate les messages pour l'API Copilot."""
    return [
        {
            "role": "system",
            "content": "Tu es un assistant spécialisé dans les GitHub actions et les workflows.",
        },
        {
            "role": "system",
            "content": f"Commence chaque réponse par le nom de l'utilisateur, qui est @{user_login}",
        },
        {
            "role": "system",
            "content": f"Utilise les informations suivantes pour enrichir ta réponse:\n\n{context}",
        },
        {
            "role": "system",
            "content": "Pour chaque réponse, termine par une section 'Le savais-tu?' qui apporte une information technique pertinente.",
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
            data = await response.json()

            if not data.get("choices") or "message" not in data["choices"][0]:
                raise ValueError("Format de réponse inattendu")

            return data["choices"][0]["message"]["content"]

    except httpx.HTTPError as http_err:
        handle_copilot_api_error(http_err)
    except Exception as e:
        logger.error("Erreur lors de l'appel à l'API Copilot: %s", e)
        raise HTTPException(
            status_code=500, detail="Erreur lors de l'appel au service Copilot"
        ) from e


def handle_copilot_api_error(error: httpx.HTTPError) -> None:
    """Gère les erreurs spécifiques de l'API Copilot."""
    if error.response.status_code == 400:
        raise HTTPException(
            status_code=400, detail="Format de requête incorrect pour l'API Copilot"
        )
    elif error.response.status_code == 401:
        raise HTTPException(
            status_code=401,
            detail="Token d'authentification Copilot invalide ou expiré",
        )
    else:
        raise HTTPException(
            status_code=error.response.status_code,
            detail=f"Erreur Copilot: {error.response.text}",
        )


async def generate_streaming_response(request_data: dict, auth_token: str):
    """Génère une réponse en streaming depuis l'API Copilot."""
    headers = {
        "authorization": f"Bearer {auth_token}",
        "content-type": "application/json",
    }
    payload = {"messages": request_data["messages"], "stream": True}
    
    async def stream_response():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                settings.COPILOT_API_URL,
                headers=headers,
                json=payload,
                timeout=None,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return stream_response()
