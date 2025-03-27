"""Service de gestion des interactions avec l'API Copilot."""

import logging
from http import HTTPStatus
from typing import Any

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
            user_data = response.json()
            return user_data.get("login")
        except Exception as e:
            logger.error("Erreur lors de l'authentification GitHub: %s", e)
            raise HTTPException(status_code=401, detail="Token GitHub invalide") from e


def prioritize_context(context_docs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pondère et ordonne les documents contextuels par pertinence.

    Args:
        context_docs: Liste des documents de contexte avec leurs scores de distance

    Returns:
        Liste triée et pondérée des documents de contexte
    """
    # Trier par score de distance (plus petit = plus proche)
    sorted_docs = sorted(context_docs, key=lambda x: x.get("distance", float("inf")))

    # Limiter au nombre de documents les plus pertinents
    max_docs = min(len(sorted_docs), 5)
    relevant_docs = sorted_docs[:max_docs]

    return relevant_docs


def extract_key_facts(
    context_docs: list[dict[str, Any]], max_facts: int = 10
) -> list[str]:
    """Extrait les informations clés des documents de contexte.

    Args:
        context_docs: Liste des documents de contexte
        max_facts: Nombre maximum de faits à extraire

    Returns:
        Liste des faits clés extraits
    """
    key_facts = []
    for doc in context_docs:
        content = doc.get("content", "")
        if not content:
            continue

        # Extraire des phrases pertinentes (simpliste pour l'instant)
        sentences = [s.strip() for s in content.split(".") if len(s.strip()) > 20]
        for sentence in sentences:
            if len(key_facts) >= max_facts:
                break
            # Éviter les doublons
            if sentence not in key_facts:
                key_facts.append(sentence)

    return key_facts


def format_structured_context(
    context_docs: list[dict[str, Any]], key_facts: list[str]
) -> str:
    """Formate le contexte de façon structurée pour le LLM.

    Args:
        context_docs: Documents de contexte
        key_facts: Faits clés extraits

    Returns:
        Contexte formaté sous forme de chaîne
    """
    if not context_docs:
        return "Aucune information contextuelle disponible."

    # Section 1: Faits clés
    formatted_context = "## Faits clés\n"
    for i, fact in enumerate(key_facts, 1):
        formatted_context += f"{i}. {fact}\n"

    # Section 2: Sources documentaires
    formatted_context += "\n## Sources détaillées\n"
    for i, doc in enumerate(context_docs, 1):
        # Extraire les métadonnées utiles
        metadata = doc.get("metadata", {})
        source = metadata.get("source", "Source inconnue")
        title = metadata.get("title", "Sans titre")

        # Ajouter un extrait de contenu
        content_preview = doc.get("content", "")[:200] + "..."
        formatted_context += f"\n### Source {i}: {title} ({source})\n"
        formatted_context += f"{content_preview}\n"

    return formatted_context


def format_copilot_messages(
    query: str, context: list[dict[str, Any]], user_login: str
) -> list[dict]:
    """Formate les messages pour l'API Copilot avec un contexte mieux structuré.

    Args:
        query: La requête de l'utilisateur
        context: Liste des documents contextuels
        user_login: Identifiant de l'utilisateur GitHub

    Returns:
        Liste de messages formatés pour l'API Copilot
    """
    # Traiter et structurer le contexte
    weighted_context = prioritize_context(context)
    key_facts = extract_key_facts(weighted_context)
    formatted_context = format_structured_context(weighted_context, key_facts)

    return [
        {
            "role": "system",
            "content": "Tu es un assistant spécialisé dans les GitHub actions et les workflows.",
        },
        {
            "role": "system",
            "content": f"Réponds de façon claire et concise à @{user_login}.",
        },
        {
            "role": "system",
            "content": f"Utilise ces informations factuelles pour répondre:\n\n{formatted_context}",
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
