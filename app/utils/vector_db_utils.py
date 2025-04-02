"""Module utilitaire pour la préparation des documents pour les bases vectorielles.
Contient des fonctions pour formater les documents pour FAISS.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _segment_non_markdown(text: str, max_length: int) -> list[str]:
    """Segmente le texte non-markdown en respectant la longueur maximale.

    Args:
        text: Texte à segmenter
        max_length: Longueur maximale de chaque segment

    Returns:
        list: Liste des segments de texte
    """
    if not text.strip():
        return []

    segments = []
    current_segment = []
    current_length = 0

    # Diviser par lignes tout en préservant les sauts de ligne
    lines = text.splitlines(keepends=True)

    for line in lines:
        line_length = len(line)

        # Si la ligne est plus longue que max_length, la découper
        if line_length > max_length:
            # Ajouter le segment en cours s'il existe
            if current_segment:
                segments.append("".join(current_segment))
                current_segment = []
                current_length = 0

            # Découper la longue ligne en segments
            while line:
                segments.append(line[:max_length])
                line = line[max_length:]

        # Si l'ajout de la ligne dépasse max_length
        elif current_length + line_length > max_length:
            segments.append("".join(current_segment))
            current_segment = [line]
            current_length = line_length

        # Sinon, ajouter la ligne au segment en cours
        else:
            current_segment.append(line)
            current_length += line_length

    # Ajouter le dernier segment s'il existe
    if current_segment:
        segments.append("".join(current_segment))

    # Ne pas appliquer strip() pour préserver le format exact
    return [s for s in segments if s]


def process_documents_for_faiss(
    documents: list[tuple[str, str]], max_length: int = 1000
) -> list[dict]:
    """Prépare les documents Markdown pour FAISS.

    Args:
        documents: Liste de tuples (chemin de fichier, contenu)
        max_length: Longueur maximale de chaque segment

    Returns:
        list: Liste de dictionnaires avec numeric_id, texte et métadonnées
    """
    from app.utils.markdown_utils import segment_text

    processed = []
    current_id = 0

    for file_path, content in documents:
        if not content.strip():
            continue

        # Utiliser segment_text uniquement pour les fichiers markdown
        if Path(file_path).suffix.lower() == ".md":
            segments = segment_text(content, max_length)
        else:
            segments = _segment_non_markdown(content, max_length)

        for idx, segment in enumerate(segments):
            if not segment:  # Ignorer les segments vides
                continue

            entry = {
                "numeric_id": current_id,
                "text": segment,  # Ne pas appliquer strip()
                "metadata": {
                    "original_id": f"{Path(file_path).stem}_{idx}",
                    "file_path": file_path,
                    "segment_index": idx,
                    "segment_id": idx,
                    "content": segment,  # Ne pas appliquer strip()
                },
            }
            processed.append(entry)
            current_id += 1

    logger.info("Nombre total de segments générés pour FAISS: %d", len(processed))
    return processed


def process_files_for_faiss(
    documents: list[tuple[str, str]], max_length: int = 1000
) -> list[dict]:
    """Prépare tous types de documents (code, markdown, texte) pour FAISS.
    Version étendue de process_documents_for_faiss prenant en charge différents types de fichiers.

    Args:
        documents: Liste de tuples (chemin de fichier, contenu)
        max_length: Longueur maximale de chaque segment

    Returns:
        list: Liste de dictionnaires avec numeric_id, texte et métadonnées
    """
    from app.utils.markdown_utils import segment_text

    processed = []
    current_id = 0

    for file_path, content in documents:
        if not content.strip():
            continue

        file_type = Path(file_path).suffix.lower()

        # Segmentation adaptée au type de fichier
        if file_type == ".md":
            segments = segment_text(content, max_length)
        else:
            segments = _segment_non_markdown(content, max_length)

        for idx, segment in enumerate(segments):
            if not segment:  # Ignorer les segments vides
                continue

            entry = {
                "numeric_id": current_id,
                "text": segment,  # Ne pas appliquer strip()
                "metadata": {
                    "original_id": f"{Path(file_path).stem}_{idx}",
                    "file_path": file_path,
                    "file_type": file_type,
                    "segment_index": idx,
                    "segment_id": idx,
                    "content": segment,  # Ne pas appliquer strip()
                },
            }
            processed.append(entry)
            current_id += 1

    logger.info("Nombre total de segments générés pour FAISS: %d", len(processed))
    return processed
