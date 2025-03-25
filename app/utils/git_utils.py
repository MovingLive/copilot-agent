"""Module utilitaire pour la gestion des dépôts Git.

Contient des fonctions pour cloner et mettre à jour des dépôts Git.
"""

import logging
import os
import subprocess
import sys
import tempfile

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


def _get_writable_directory(target_dir: str) -> str:
    """Trouve un répertoire accessible en écriture."""
    try:
        configured_dir = os.path.abspath(target_dir)
        parent_dir = os.path.dirname(configured_dir)

        if not os.path.exists(parent_dir):
            try:
                os.makedirs(parent_dir, exist_ok=True)
            except OSError as e:
                if e.errno == 30:  # EROFS - Read-only file system
                    logger.warning(
                        "Système de fichiers en lecture seule détecté pour %s",
                        parent_dir,
                    )
                    return _create_temp_directory()
                logger.warning(
                    "Impossible de créer le répertoire parent %s: %s", parent_dir, e
                )
                return _create_temp_directory()

        if os.path.exists(configured_dir):
            if _is_directory_writable(configured_dir):
                return configured_dir
            return _create_temp_directory()

        try:
            os.makedirs(configured_dir, exist_ok=True)
            return configured_dir
        except OSError:
            return _create_temp_directory()

    except OSError as e:
        logger.warning("Erreur lors de la vérification des permissions: %s", e)
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
    """Clone ou met à jour un dépôt Git."""
    try:
        if not os.path.exists(os.path.join(repo_dir, ".git")):
            logger.info(
                "Clonage du repo depuis %s dans le dossier %s...", repo_url, repo_dir
            )
            subprocess.run(["git", "clone", repo_url, repo_dir], check=True)
        else:
            logger.info("Mise à jour du repo dans le dossier %s...", repo_dir)
            subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
        return repo_dir
    except subprocess.CalledProcessError:
        logger.error("Erreur lors de l'opération Git.")
        sys.exit(1)
