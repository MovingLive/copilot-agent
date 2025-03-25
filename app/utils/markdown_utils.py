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
            if current_segment and len(current_segment) > 10:
                segments.append(current_segment.strip())
            current_segment = line + "\n"
            current_title = line
        # Si l'ajout de la ligne dépasse max_length
        elif len(current_segment) + len(line) > max_length:
            # Ajouter le segment courant
            segments.append(current_segment.strip())
            # Commencer un nouveau segment avec le titre et la ligne courante
            new_segment = current_title + "\n" + line + "\n"
            # Si le nouveau segment est déjà trop long, le diviser
            if len(new_segment) > max_length:
                # Ajouter le titre seul
                segments.append(current_title.strip())
                # Diviser la ligne en segments de max_length
                remaining = line
                while remaining:
                    segment_length = max_length - len(current_title) - 2  # -2 pour \n
                    segments.append(current_title + "\n" + remaining[:segment_length])
                    remaining = remaining[segment_length:]
                current_segment = current_title + "\n"
            else:
                current_segment = new_segment
        else:
            current_segment += line + "\n"

    # Ajouter le dernier segment s'il n'est pas vide
    if current_segment and len(current_segment) > 10:
        segments.append(current_segment.strip())

    return segments
