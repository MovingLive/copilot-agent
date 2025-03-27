"""Module utilitaire pour la gestion des dépôts Git.

Contient des fonctions pour cloner et mettre à jour des dépôts Git.
"""

import logging
import os
import re
import subprocess
import sys
import tempfile

# Errno constant for read-only file system
EROFS = 30

logger = logging.getLogger(__name__)


def clone_or_update_repo(repo_url: str, repo_dir: str) -> str:
    """Clone ou met à jour le dépôt GitHub contenant la documentation.

    Utilise un répertoire temporaire si le répertoire cible n'est pas accessible en écriture.

    Args:
        repo_url: URL du dépôt GitHub
        repo_dir: Dossier local cible pour le clonage/mise à jour

    Returns:
        str: Chemin du répertoire contenant le dépôt cloné
    """
    # Vérifier si nous sommes en environnement de test et le type de test
    is_testing = os.getenv("TESTING", "false").lower() == "true"
    skip_git_calls = os.getenv("SKIP_GIT_CALLS", "false").lower() == "true"

    # Utiliser le répertoire défini ou créer un répertoire temporaire
    repo_dir_path = _get_writable_directory(repo_dir)

    # Ne pas exécuter les commandes git en mode test si SKIP_GIT_CALLS est activé
    if is_testing and skip_git_calls:
        logger.info("Mode test avec simulation du clonage détecté: simulation du dépôt")
        _create_test_document(repo_dir_path)
        return repo_dir_path

    # Cloner ou mettre à jour le dépôt
    return _clone_or_pull_repo(repo_url, repo_dir_path)


def _handle_parent_directory(parent_dir: str, configured_dir: str) -> str:
    """Gère la création du répertoire parent."""
    try:
        os.makedirs(parent_dir, exist_ok=True)
        return configured_dir
    except OSError as e:
        if e.errno == EROFS:  # Read-only file system
            logger.warning(
                "Système de fichiers en lecture seule détecté pour %s", parent_dir
            )
        else:
            logger.warning(
                "Impossible de créer le répertoire parent %s: %s", parent_dir, e
            )
        return _create_temp_directory()


def _get_writable_directory(target_dir: str) -> str:
    """Trouve un répertoire accessible en écriture."""
    try:
        configured_dir = os.path.abspath(target_dir)
        parent_dir = os.path.dirname(configured_dir)

        if not os.path.exists(parent_dir):
            return _handle_parent_directory(parent_dir, configured_dir)

        return _handle_configured_directory(configured_dir)

    except OSError as e:
        logger.warning("Erreur lors de la vérification des permissions: %s", e)
        return _create_temp_directory()


def _handle_configured_directory(configured_dir: str) -> str:
    """Gère la vérification et création du répertoire configuré."""
    if os.path.exists(configured_dir):
        return (
            configured_dir
            if _is_directory_writable(configured_dir)
            else _create_temp_directory()
        )

    try:
        os.makedirs(configured_dir, exist_ok=True)
        return configured_dir
    except OSError:
        return _create_temp_directory()


def _create_temp_directory() -> str:
    """Crée et retourne un répertoire temporaire."""
    temp_dir = os.path.join(tempfile.gettempdir(), "repo_clone_" + str(os.getpid()))
    os.makedirs(temp_dir, exist_ok=True)
    logger.info("Utilisation du répertoire temporaire: %s", temp_dir)
    return temp_dir


def _is_directory_writable(directory: str) -> bool:
    """Vérifie si un répertoire est accessible en écriture."""
    try:
        test_file = os.path.join(directory, ".write_test")
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except OSError:
        return False


def _create_test_document(directory: str) -> None:
    """Crée un document de test pour les tests unitaires."""
    test_file = os.path.join(directory, "test_document.md")
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(
            "# Test Document\n\nCeci est un document de test pour les tests unitaires.\n"
        )


def _clone_or_pull_repo(repo_url: str, repo_dir: str) -> str:
    """Clone ou met à jour un dépôt Git.

    Supporte l'authentification pour les dépôts privés via:
    - GitHub Personal Access Token (PAT)
    - GitHub App (JWT)
    - GitHub Actions token

    Args:
        repo_url: URL du dépôt Git
        repo_dir: Répertoire local pour le clonage ou la mise à jour

    Returns:
        str: Chemin du répertoire contenant le dépôt cloné

    Raises:
        SystemExit: Si l'opération Git échoue
    """
    try:
        # Vérifier si c'est une URL GitHub et configurer l'authentification si nécessaire
        auth_url = repo_url
        auth_message = None
        if _is_github_url(repo_url):
            auth_url, auth_message = _get_github_auth_url(repo_url)
            if auth_message:
                logger.info(auth_message)

        if not os.path.exists(os.path.join(repo_dir, ".git")):
            logger.info(
                "Clonage du repo depuis %s dans le dossier %s...",
                repo_url.split("@")[-1]
                if "@" in repo_url
                else repo_url,  # Masquer les identifiants
                repo_dir,
            )
            subprocess.run(["git", "clone", auth_url, repo_dir], check=True)
        else:
            logger.info("Mise à jour du repo dans le dossier %s...", repo_dir)
            # Mettre à jour l'URL distante si nécessaire pour l'authentification
            if auth_message:
                subprocess.run(
                    ["git", "-C", repo_dir, "remote", "set-url", "origin", auth_url],
                    check=True,
                )
            subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
        return repo_dir
    except subprocess.CalledProcessError as e:
        logger.error("Erreur lors de l'opération Git: %s", e)
        sys.exit(1)


def _is_github_url(url: str) -> bool:
    """Vérifie si l'URL est une URL GitHub.

    Args:
        url: L'URL à vérifier

    Returns:
        bool: True si l'URL est une URL GitHub
    """
    return bool(re.match(r"^https?://(?:www\.)?github\.com/", url))


def _get_github_auth_url(repo_url: str) -> tuple[str, str | None]:
    """Construit une URL authentifiée pour GitHub si les identifiants sont disponibles.

    Args:
        repo_url: URL du dépôt GitHub

    Returns:
        Tuple[str, Optional[str]]: URL modifiée et message d'authentification
    """
    # Vérifier si GitHub Actions est utilisé
    if os.getenv("GITHUB_ACTIONS") == "true":
        github_token = os.getenv("GITHUB_TOKEN")
        if github_token:
            # Dans GitHub Actions, utiliser le token GITHUB_TOKEN
            auth_message = "Authentification via GITHUB_TOKEN dans GitHub Actions"
            return _add_token_to_url(repo_url, github_token), auth_message

    # Vérifier si un GitHub App est configuré
    github_app_id = os.getenv("GITHUB_APP_ID")
    github_app_private_key = os.getenv("GITHUB_APP_PRIVATE_KEY")
    if github_app_id and github_app_private_key:
        # Utiliser GitHub App pour l'authentification
        try:
            from datetime import datetime, timedelta

            import jwt

            # Créer un token JWT pour l'authentification GitHub App
            now = int(datetime.now().timestamp())
            payload = {
                "iat": now,  # Issued at time
                "exp": now + 600,  # Expiration time (10 minutes)
                "iss": github_app_id,  # GitHub App ID
            }

            # Encoder la clé privée
            private_key = github_app_private_key.replace("\\n", "\n")
            encoded_jwt = jwt.encode(payload, private_key, algorithm="RS256")

            auth_message = "Authentification via GitHub App"
            return _add_token_to_url(repo_url, encoded_jwt), auth_message
        except (ImportError, Exception) as e:
            logger.warning("Erreur lors de l'authentification GitHub App: %s", e)

    # Si un PAT est configuré, l'utiliser
    github_token = os.getenv("GITHUB_PAT")
    if github_token:
        auth_message = "Authentification via GitHub Personal Access Token"
        return _add_token_to_url(repo_url, github_token), auth_message

    # Aucune authentification trouvée
    return repo_url, None


def _add_token_to_url(repo_url: str, token: str) -> str:
    """Ajoute un token à l'URL GitHub.

    Args:
        repo_url: URL du dépôt GitHub
        token: Token d'authentification

    Returns:
        str: URL avec token d'authentification
    """
    # Format: https://username:token@github.com/owner/repo.git
    url_parts = repo_url.split("://", 1)
    if len(url_parts) != 2:
        return repo_url

    protocol, rest = url_parts
    return f"{protocol}://x-access-token:{token}@{rest}"
