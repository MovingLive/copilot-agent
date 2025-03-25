"""Module utilitaire pour le traitement des documents Markdown.
Contient des fonctions pour lire et segmenter des fichiers Markdown.
"""

import glob
import logging
import os

logger = logging.getLogger(__name__)


def read_markdown_files(repo_dir: str) -> list[tuple[str, str]]:
    """Lit tous les fichiers Markdown du dépôt.

    Args:
        repo_dir: Chemin du répertoire contenant les fichiers Markdown

    Returns:
        list: Liste de tuples (chemin de fichier, contenu)
    """
    markdown_files = glob.glob(os.path.join(repo_dir, "**", "*.md"), recursive=True)
    documents = []

    for file_path in markdown_files:
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()
                documents.append((file_path, content))
        except OSError as e:
            logger.warning("Impossible de lire le fichier %s: %s", file_path, e)

    logger.info("Nombre de fichiers Markdown lus: %d", len(documents))
    return documents


def segment_text(text: str, max_length: int = 1000) -> list[str]:
    """Segmente le texte en morceaux de longueur maximale max_length.

    Args:
        text: Texte à segmenter
        max_length: Longueur maximale de chaque segment

    Returns:
        list: Liste des segments de texte
    """
    segments = []
    lines = text.split("\n")
    current_segment = ""
    current_title = ""

    for line in lines:
        # Détection des titres (## ou ###)
        is_title = line.strip().startswith("##")

        if is_title:
            # Si on a un segment en cours, on l'ajoute
            if current_segment:
                segments.append(current_segment.rstrip())
            current_title = line
            current_segment = current_title + "\n"
        else:
            # Gérer les lignes vides
            if not line.strip():
                if len(current_segment) + 1 <= max_length:
                    current_segment += "\n"
                continue

            # Si la ligne est trop longue, on la divise
            while line:
                available_space = max_length - len(current_segment)
                if available_space <= 0:
                    # Le segment courant est plein, on l'ajoute et on en commence un nouveau
                    segments.append(current_segment.rstrip())
                    current_segment = current_title + "\n"
                    available_space = max_length - len(current_segment)

                # On prend autant de caractères que possible
                chunk = line[:available_space]
                current_segment += chunk
                line = line[available_space:]  # Le reste pour la prochaine itération

                if line:  # S'il reste du texte à traiter
                    current_segment = current_segment.rstrip() + "\n"

            current_segment = current_segment.rstrip() + "\n"

    # Ajouter le dernier segment s'il n'est pas vide
    if current_segment:
        segments.append(current_segment.rstrip())

    return segments
