"""Module utilitaire pour la préparation des documents pour les bases vectorielles.
Contient des fonctions pour formater les documents pour FAISS.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


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
        segments = segment_text(content, max_length)
        for idx, segment in enumerate(segments):
            entry = {
                "numeric_id": current_id,
                "text": segment,
                "metadata": {
                    "original_id": f"{Path(file_path).stem}_{idx}",
                    "file_path": file_path,
                    "segment_index": idx,
                    "segment_id": idx,
                    "content": segment,
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
        # Déterminer le type de fichier
        file_type = Path(file_path).suffix.lower()

        # Segmenter le contenu
        segments = segment_text(content, max_length)

        for idx, segment in enumerate(segments):
            entry = {
                "numeric_id": current_id,
                "text": segment,
                "metadata": {
                    "original_id": f"{Path(file_path).stem}_{idx}",
                    "file_path": file_path,
                    "file_type": file_type,
                    "segment_index": idx,
                    "segment_id": idx,
                    "content": segment,
                },
            }
            processed.append(entry)
            current_id += 1

    logger.info("Nombre total de segments générés pour FAISS: %d", len(processed))
    return processed
