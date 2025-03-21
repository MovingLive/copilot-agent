"""
Module utilitaire pour la gestion des dépôts Git.
Contient des fonctions pour cloner, mettre à jour et manipuler des dépôts Git.
"""

import logging
import os
import subprocess
import sys
import tempfile


def clone_or_update_repo(repo_url: str, repo_dir: str) -> str:
    """
    Clone ou met à jour le dépôt GitHub contenant la documentation.
    Utilise un répertoire temporaire si le répertoire cible n'est pas accessible en écriture.

    Args:
        repo_url: URL du dépôt GitHub
        repo_dir: Dossier local cible pour le clonage/mise à jour

    Returns:
        str: Chemin du répertoire contenant le dépôt cloné
    """
    configured_dir = os.path.abspath(repo_dir)
    repo_dir_path = configured_dir
    
    try:
        # Tester si on peut écrire dans le répertoire parent
        parent_dir = os.path.dirname(configured_dir)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)
        
        # Si le répertoire existe, on vérifie qu'on peut y écrire
        if os.path.exists(configured_dir):
            test_file = os.path.join(configured_dir, ".write_test")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
            except (IOError, OSError):
                # Si on ne peut pas écrire, on utilise un répertoire temporaire
                repo_dir_path = os.path.join(
                    tempfile.gettempdir(),
                    f"repo_clone_{os.getpid()}"
                )
                os.makedirs(repo_dir_path, exist_ok=True)
        else:
            # Si le répertoire n'existe pas, on essaie de le créer
            try:
                os.makedirs(configured_dir, exist_ok=True)
            except (IOError, OSError):
                # Si on ne peut pas créer le répertoire, on utilise un temporaire
                repo_dir_path = os.path.join(
                    tempfile.gettempdir(),
                    f"repo_clone_{os.getpid()}"
                )
                os.makedirs(repo_dir_path, exist_ok=True)

    except Exception as e:
        logging.warning("Erreur lors de la vérification des permissions: %s", e)
        # Utiliser un répertoire temporaire comme fallback
        repo_dir_path = os.path.join(
            tempfile.gettempdir(),
            f"repo_clone_{os.getpid()}"
        )
        os.makedirs(repo_dir_path, exist_ok=True)

    # Cloner ou mettre à jour le dépôt
    git_dir = os.path.join(repo_dir_path, ".git")
    if not os.path.exists(git_dir):
        logging.info(
            "Clonage du repo depuis %s dans le dossier %s...", repo_url, repo_dir_path
        )
        try:
            subprocess.run(["git", "clone", repo_url, repo_dir_path], check=True)
        except subprocess.CalledProcessError:
            logging.error("Erreur lors du clonage du repo.")
            sys.exit(1)
    else:
        logging.info("Mise à jour du repo dans le dossier %s...", repo_dir_path)
        try:
            subprocess.run(["git", "-C", repo_dir_path, "pull"], check=True)
        except subprocess.CalledProcessError:
            logging.error("Erreur lors de la mise à jour du repo.")
            sys.exit(1)

    return repo_dir_path
