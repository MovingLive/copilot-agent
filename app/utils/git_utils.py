"""Module utilitaire pour la gestion des dépôts Git via GitHub API.

Contient des fonctions pour lire le contenu des dépôts GitHub via l'API.
"""

import base64
import logging
import os
from pathlib import Path

from github import Github, GithubException
from github.Repository import Repository

from app.utils.document_utils import EXCLUDED_EXTENSIONS, INCLUDED_CODE_EXTENSIONS

logger = logging.getLogger(__name__)


def get_github_client() -> Github:
    """Crée un client GitHub authentifié.

    Returns:
        Github: Instance authentifiée du client GitHub
    """
    # Vérifier si GitHub Actions est utilisé
    if os.getenv("GITHUB_ACTIONS") == "true" and os.getenv("GITHUB_TOKEN"):
        token = os.getenv("GITHUB_TOKEN")
        logger.info("Utilisation de l'authentification GitHub Actions")
        return Github(token, timeout=30)

    # Vérifier si un PAT est configuré
    if os.getenv("GITHUB_PAT"):
        token = os.getenv("GITHUB_PAT")
        logger.info("Utilisation du Personal Access Token GitHub")
        return Github(token, timeout=30)

    # Si pas d'authentification, utiliser un client non authentifié
    logger.warning(
        "Aucune authentification GitHub configurée. Les limites de taux seront strictes (60 requêtes/heure)"
    )
    return Github(timeout=30)


def is_file_relevant(file_path: str) -> bool:
    """Détermine si un fichier doit être lu en fonction de son extension.

    Args:
        file_path: Chemin du fichier à vérifier

    Returns:
        bool: True si le fichier doit être lu, False sinon
    """
    ext = Path(file_path).suffix.lower()

    # Vérifier d'abord si l'extension est dans la liste des extensions exclues
    if ext in EXCLUDED_EXTENSIONS:
        return False

    # Vérifier si l'extension est dans la liste des extensions incluses
    if ext in INCLUDED_CODE_EXTENSIONS:
        return True

    return False


def read_file_content(repo: Repository, file_path: str) -> str | None:
    """Lit le contenu d'un fichier depuis un repository GitHub.

    Args:
        repo: Repository GitHub
        file_path: Chemin du fichier dans le repository

    Returns:
        Optional[str]: Contenu du fichier ou None si erreur ou fichier non pertinent
    """
    try:
        # Vérifier d'abord si le fichier est pertinent
        if not is_file_relevant(file_path):
            logger.debug("Le fichier %s n'est pas pertinent pour l'indexation", file_path)
            return None

        content = repo.get_contents(file_path)
        if isinstance(content, list):
            logger.debug("Le chemin %s est un dossier", file_path)
            return None

        try:
            decoded_content = base64.b64decode(content.content).decode("utf-8")
            if not decoded_content.strip():
                logger.debug("Le fichier %s est vide", file_path)
                return None
            return decoded_content
        except UnicodeDecodeError:
            logger.debug("Le fichier %s n'est pas un fichier texte valide", file_path)
            return None

    except GithubException as e:
        if e.status == 403 and "rate limit exceeded" in str(e.data.get("message", "")):
            logger.error(
                "Limite de taux GitHub dépassée lors de la lecture de %s. "
                "Utilisez GITHUB_PAT pour augmenter la limite.",
                file_path,
            )
        else:
            logger.warning("Impossible de lire le fichier %s: %s", file_path, e)
        return None


def get_repository(owner: str, repo_name: str) -> Repository | None:
    """Récupère un repository GitHub.

    Args:
        owner: Propriétaire du repository
        repo_name: Nom du repository

    Returns:
        Optional[Repository]: Instance du repository ou None si non trouvé
    """
    try:
        client = get_github_client()
        repo = client.get_repo(f"{owner}/{repo_name}")
        return repo
    except GithubException as e:
        logger.error("Erreur lors de l'accès au repository %s/%s: %s", owner, repo_name, e)
        return None


def list_repository_files(repo: Repository, path: str = "") -> list[tuple[str, str]]:
    """Liste tous les fichiers d'un repository GitHub.

    Args:
        repo: Repository GitHub
        path: Chemin dans le repository (optionnel)

    Returns:
        list[tuple[str, str]]: Liste de tuples (chemin du fichier, contenu)
    """
    files = []
    try:
        contents = repo.get_contents(path)
        files_count = 0

        while contents:
            content = contents.pop(0)
            if files_count % 10 == 0:  # Log tous les 10 fichiers
                logger.info("Progression: %d fichiers traités", files_count)

            if content.type == "dir":
                try:
                    dir_contents = repo.get_contents(content.path)
                    contents.extend(dir_contents)
                except GithubException as e:
                    logger.warning(
                        "Erreur lors de la lecture du dossier %s: %s", content.path, e
                    )
            else:
                file_content = read_file_content(repo, content.path)
                if file_content is not None:
                    files.append((content.path, file_content))
                files_count += 1

        logger.info("Total: %d fichiers traités", files_count)

    except GithubException as e:
        if e.status == 403 and "rate limit exceeded" in str(e.data.get("message", "")):
            logger.error(
                "Limite de taux GitHub dépassée lors du listage des fichiers. "
                "Utilisez GITHUB_PAT pour augmenter la limite."
            )
        else:
            logger.error("Erreur lors de la lecture du repository: %s", e)

    return files


def read_repository_content(owner_repo: str) -> list[tuple[str, str]]:
    """Lit tout le contenu pertinent d'un repository GitHub.

    Args:
        owner_repo: Repository au format "owner/repo"

    Returns:
        list[tuple[str, str]]: Liste de tuples (chemin du fichier, contenu)
    """
    try:
        owner, repo_name = owner_repo.split("/")
    except ValueError:
        logger.error("Format de repository invalide. Utiliser 'owner/repo': %s", owner_repo)
        return []

    repo = get_repository(owner, repo_name)
    if not repo:
        return []

    return list_repository_files(repo)
