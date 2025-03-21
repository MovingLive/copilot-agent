"""
Module utilitaire pour le traitement de documents Markdown.
Contient des fonctions pour lire, segmenter et traiter des fichiers Markdown.
"""

import glob
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path


# --- Fonctions utilitaires ---
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
    # Vérifier si nous sommes en environnement de test et le type de test
    is_testing = os.getenv("TESTING", "false").lower() == "true"
    skip_git_calls = os.getenv("SKIP_GIT_CALLS", "false").lower() == "true"

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

    # Ne pas exécuter les commandes git en mode test si SKIP_GIT_CALLS est activé
    if is_testing and skip_git_calls:
        logging.info(
            "Mode test avec simulation du clonage détecté: simulation du dépôt"
        )
        # Créer un fichier markdown factice pour les tests
        test_file = os.path.join(repo_dir_path, "test_document.md")
        with open(test_file, "w") as f:
            f.write(
                "# Test Document\n\nCeci est un document de test pour les tests unitaires.\n"
            )
        return repo_dir_path

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


def read_markdown_files(repo_dir: str) -> list[tuple[str, str]]:
    """
    Lit tous les fichiers Markdown du dépôt et retourne une liste de tuples (chemin, contenu).

    Args:
        repo_dir: Chemin du répertoire contenant les fichiers Markdown

    Returns:
        list: Liste de tuples (chemin de fichier, contenu)
    """
    markdown_files = glob.glob(os.path.join(repo_dir, "**", "*.md"), recursive=True)
    documents = []
    for file_path in markdown_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
                documents.append((file_path, content))
        except (IOError, OSError) as e:
            logging.warning("Impossible de lire le fichier %s: %s", file_path, e)
    logging.info("Nombre de fichiers Markdown lus: %d", len(documents))
    return documents


def segment_text(text: str, max_length: int = 500) -> list[str]:
    """
    Segmente le texte en morceaux de longueur maximale max_length.
    Une segmentation simple basée sur les sauts de ligne et la longueur.

    Args:
        text: Texte à segmenter
        max_length: Longueur maximale de chaque segment

    Returns:
        list: Liste des segments de texte
    """
    segments = []
    paragraphs = text.split("\n\n")
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # Si le paragraphe est trop long, le découper en morceaux
        if len(para) > max_length:
            for i in range(0, len(para), max_length):
                segments.append(para[i : i + max_length])
        else:
            segments.append(para)
    return segments


def process_documents_for_chroma(
    documents: list[tuple[str, str]], max_length: int = 500
) -> list[dict]:
    """
    Pour chaque document, segmente le contenu et retourne une liste de dictionnaires contenant
    le texte, un identifiant et des métadonnées pour ChromaDB.

    Args:
        documents: Liste de tuples (chemin de fichier, contenu)
        max_length: Longueur maximale de chaque segment

    Returns:
        list: Liste de dictionnaires avec id, texte et métadonnées
    """
    processed = []
    for file_path, content in documents:
        segments = segment_text(content, max_length)
        for idx, segment in enumerate(segments):
            document_id = f"{Path(file_path).stem}_{idx}"
            entry = {
                "id": document_id,
                "text": segment,
                "metadata": {"file_path": file_path, "segment_index": idx},
            }
            processed.append(entry)
    logging.info("Nombre total de segments générés: %d", len(processed))
    return processed


def process_documents_for_faiss(
    documents: list[tuple[str, str]], max_length: int = 500
) -> list[dict]:
    """
    Pour chaque document, segmente le contenu et retourne une liste de dictionnaires contenant
    le texte, un identifiant numérique et des métadonnées pour FAISS.

    Args:
        documents: Liste de tuples (chemin de fichier, contenu)
        max_length: Longueur maximale de chaque segment

    Returns:
        list: Liste de dictionnaires avec numeric_id, texte et métadonnées
    """
    processed = []
    current_id = 0
    for file_path, content in documents:
        segments = segment_text(content, max_length)
        for idx, segment in enumerate(segments):
            entry = {
                "numeric_id": current_id,
                "text": segment,
                "metadata": {
                    "original_id": f"{Path(file_path).stem}_{idx}",
                    "file_path": file_path,
                    "segment_index": idx,
                },
            }
            processed.append(entry)
            current_id += 1
    logging.info("Nombre total de segments générés: %d", len(processed))
    return processed
