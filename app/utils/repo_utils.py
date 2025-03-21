"""
Module utilitaire pour la gestion des dépôts Git.
Contient des fonctions pour cloner, mettre à jour et manipuler des dépôts Git.
"""

import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path


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
    # Utiliser le répertoire défini par la variable d'environnement ou créer un répertoire temporaire
    repo_dir_path = None

    # D'abord, essayer le répertoire configuré
    configured_dir = os.path.abspath(repo_dir)

    try:
        # Tester si on peut écrire dans le répertoire parent
        parent_dir = os.path.dirname(configured_dir)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        # Si le répertoire existe déjà, vérifier qu'on peut y écrire
        if os.path.exists(configured_dir):
            test_file = os.path.join(configured_dir, ".write_test")
            try:
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                repo_dir_path = configured_dir
            except (IOError, OSError):
                logging.warning(
                    "Le répertoire %s n'est pas accessible en écriture", configured_dir
                )
        else:
            # Essayer de créer le répertoire
            try:
                os.makedirs(configured_dir, exist_ok=True)
                repo_dir_path = configured_dir
            except (IOError, OSError) as e:
                logging.warning(
                    "Impossible de créer le répertoire %s: %s", configured_dir, e
                )
    except Exception as e:
        logging.warning("Erreur lors de la vérification des permissions: %s", e)

    # Si on n'a pas pu utiliser le répertoire configuré, utiliser un répertoire temporaire
    if repo_dir_path is None:
        repo_dir_path = os.path.join(
            tempfile.gettempdir(), "repo_clone_" + str(os.getpid())
        )
        logging.info("Utilisation du répertoire temporaire: %s", repo_dir_path)
        if not os.path.exists(repo_dir_path):
            os.makedirs(repo_dir_path, exist_ok=True)

    # Cloner ou mettre à jour le dépôt
    if not os.path.exists(os.path.join(repo_dir_path, ".git")):
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
