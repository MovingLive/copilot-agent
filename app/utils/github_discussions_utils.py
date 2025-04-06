"""Module utilitaire pour l'extraction des discussions GitHub.

Ce module fournit des fonctions pour récupérer les discussions GitHub validées
via l'API GraphQL, en extrayant les titres, descriptions et réponses validées.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv

from app.core.config import settings

# Chargement des variables d'environnement
load_dotenv()

# Configuration du logger
logger = logging.getLogger(__name__)

# Configuration de l'API GitHub
GITHUB_API_URL = "https://api.github.com/graphql"


def execute_graphql_query(query: str, variables: Dict[str, Any]) -> Dict[str, Any]:
    """Exécute une requête GraphQL sur l'API GitHub.

    Args:
        query: La requête GraphQL à exécuter
        variables: Les variables à passer à la requête

    Returns:
        Dict[str, Any]: Les données retournées par l'API

    Raises:
        ValueError: Si le token GitHub n'est pas configuré
        httpx.HTTPStatusError: Si la requête échoue
    """
    # Obtenir le token du paramètre centralisé
    token = settings.GITHUB_TOKEN
    if not token:
        raise ValueError(
            "Token GitHub non configuré. Définissez GITHUB_TOKEN dans .env"
        )

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        response = httpx.post(
            GITHUB_API_URL,
            json={"query": query, "variables": variables},
            headers=headers,
            timeout=30.0,
        )
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as e:
        logger.error("Erreur lors de la requête GraphQL: %s", e)
        logger.error(
            "Détails: %s", response.text if "response" in locals() else "Pas de réponse"
        )
        raise
    except httpx.RequestError as e:
        logger.error("Erreur de requête GraphQL: %s", e)
        raise


def extract_repo_info(repo_url: str) -> Optional[Tuple[str, str]]:
    """Extrait le propriétaire et le nom du dépôt à partir d'une URL GitHub.

    Args:
        repo_url: L'URL du dépôt GitHub

    Returns:
        Tuple[str, str] ou None: Le propriétaire et le nom du dépôt, ou None si l'URL est invalide
    """
    # Supprimer l'extension .git si présente
    clean_url = repo_url.replace(".git", "")

    # Gérer les URL sans protocole (github.com/owner/repo)
    if clean_url.startswith("github.com/"):
        parts = clean_url.split("/")
        if len(parts) >= 3:
            return parts[1], parts[2].rstrip("/")

    # Traiter les URL au format https://github.com/owner/repo
    parts = clean_url.split("/")
    if len(parts) >= 4 and "github.com" in parts:
        # Trouver l'index de github.com
        github_index = parts.index("github.com")
        if len(parts) > github_index + 2:
            owner = parts[github_index + 1]
            repo = parts[github_index + 2].rstrip("/")
            return owner, repo

    return None


def fetch_validated_discussions(
    owner: str, repo: str, max_discussions: int = 100
) -> List[Dict[str, Any]]:
    """Récupère les discussions GitHub avec des réponses validées.

    Args:
        owner: Le propriétaire du dépôt
        repo: Le nom du dépôt
        max_discussions: Le nombre maximum de discussions à récupérer

    Returns:
        List[Dict[str, Any]]: Liste des discussions avec leur titre, description et réponses validées
    """
    # Requête GraphQL pour récupérer les discussions avec leur réponse validée
    query = """
    query FetchValidatedDiscussions($owner: String!, $name: String!, $first: Int!) {
      repository(owner: $owner, name: $name) {
        discussions(first: $first) {
          nodes {
            id
            title
            body
            url
            answer {
              id
              body
              author {
                login
              }
            }
            category {
              name
            }
            labels(first: 10) {
              nodes {
                name
              }
            }
          }
        }
      }
    }
    """

    variables = {
        "owner": owner,
        "name": repo,
        "first": max_discussions,
    }

    try:
        # Exécution de la requête GraphQL
        result = execute_graphql_query(query, variables)

        # Extraction des discussions avec réponses validées
        validated_discussions = []
        if (
            "data" in result
            and "repository" in result["data"]
            and "discussions" in result["data"]["repository"]
            and "nodes" in result["data"]["repository"]["discussions"]
        ):
            discussions = result["data"]["repository"]["discussions"]["nodes"]

            # Filtrer pour ne garder que les discussions avec réponses validées
            for discussion in discussions:
                if discussion["answer"] is not None:
                    validated_discussions.append(
                        {
                            "id": discussion["id"],
                            "title": discussion["title"],
                            "body": discussion["body"],
                            "url": discussion["url"],
                            "answer": discussion["answer"]["body"],
                            "answer_author": discussion["answer"]["author"]["login"],
                            "category": discussion["category"]["name"]
                            if discussion["category"]
                            else None,
                            "labels": [
                                label["name"] for label in discussion["labels"]["nodes"]
                            ]
                            if discussion["labels"]["nodes"]
                            else [],
                        }
                    )

            logger.info(
                "Récupéré %d discussions avec réponses validées sur %d discussions",
                len(validated_discussions),
                len(discussions),
            )

        return validated_discussions

    except Exception as e:
        logger.error("Erreur lors de la récupération des discussions: %s", e)
        return []


def format_discussions_for_faiss(
    discussions: List[Dict[str, Any]],
) -> List[Tuple[str, str]]:
    """Formate les discussions GitHub pour l'indexation FAISS.

    Args:
        discussions: Liste des discussions avec leurs réponses validées

    Returns:
        List[Tuple[str, str]]: Liste de tuples (file_path, content) prêts pour FAISS
    """
    formatted_discussions = []

    for idx, discussion in enumerate(discussions):
        # Générer un chemin de fichier virtuel pour chaque discussion
        file_path = f"github_discussions/{discussion['category']}/{idx}_{discussion['title'].replace(' ', '_')}.md"

        # Formater le contenu avec le titre, la description et la réponse validée
        content = f"""# {discussion["title"]}

## Description
{discussion["body"]}

## Réponse validée
{discussion["answer"]}

---
Discussion URL: {discussion["url"]}
Auteur de la réponse: {discussion["answer_author"]}
Catégorie: {discussion["category"]}
Labels: {", ".join(discussion["labels"])}
"""

        formatted_discussions.append((file_path, content))

    return formatted_discussions


def get_validated_discussions_from_repos(repo_urls: List[str]) -> List[Tuple[str, str]]:
    """Récupère les discussions validées de plusieurs dépôts GitHub.

    Args:
        repo_urls: Liste des URL des dépôts GitHub

    Returns:
        List[Tuple[str, str]]: Liste des discussions formatées pour FAISS
    """
    all_discussions = []

    for url in repo_urls:
        # Extraire le propriétaire et le nom du dépôt
        repo_info = extract_repo_info(url)
        if not repo_info:
            logger.warning("URL de dépôt invalide: %s", url)
            continue

        owner, repo = repo_info
        logger.info("Récupération des discussions validées pour %s/%s", owner, repo)

        # Récupérer les discussions validées
        discussions = fetch_validated_discussions(owner, repo)

        # Formater les discussions pour FAISS
        if discussions:
            formatted_discussions = format_discussions_for_faiss(discussions)
            all_discussions.extend(formatted_discussions)
            logger.info(
                "Ajout de %d discussions formatées pour %s/%s",
                len(formatted_discussions),
                owner,
                repo,
            )

    return all_discussions
