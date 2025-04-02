"""Module utilitaire pour le traitement des documents Markdown.

Contient des fonctions pour lire et segmenter des fichiers Markdown.
"""

import glob
import logging
import os
import re

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


def analyze_markdown_structure(text: str) -> list[dict[str, str]]:
    """Analyse la structure du document Markdown pour identifier les sections.

    Args:
        text: Texte Markdown à analyser

    Returns:
        Liste de dictionnaires contenant les titres et leur contenu
    """
    sections = []
    lines = text.split("\n")

    # Si le texte est court et contient un seul titre, le traiter comme une seule section
    if len(lines) <= 2 and any(line.startswith("#") for line in lines):
        return [{
            "title": "",
            "level": 0,
            "content": text
        }]

    current_title = None
    current_content = []

    for line in lines:
        # Détection des titres (# à ####)
        title_match = re.match(r"^(#{1,4})\s+(.+)$", line)

        if title_match:
            # Sauvegarder la section précédente
            if current_content:
                sections.append({
                    "title": current_title or "",
                    "level": current_title.count("#") if current_title else 0,
                    "content": "\n".join(current_content),
                })

            # Nouvelle section
            current_title = line
            current_content = []
        else:
            current_content.append(line)

    # Ajouter la dernière section
    if current_content:
        sections.append({
            "title": current_title or "",
            "level": current_title.count("#") if current_title else 0,
            "content": "\n".join(current_content),
        })

    # Si aucune section n'a été trouvée, retourner le texte complet comme une section
    if not sections:
        return [{
            "title": "",
            "level": 0,
            "content": text,
        }]

    return sections


def segment_by_semantic_units(
    text: str, max_length: int, overlap: int = 50
) -> list[str]:
    """Segmente le texte en unités sémantiques avec chevauchement.

    Args:
        text: Texte à segmenter
        max_length: Longueur maximale de chaque segment
        overlap: Nombre de caractères de chevauchement entre segments

    Returns:
        Liste des segments de texte
    """
    if not text.strip():
        return []

    segments = []
    paragraphs = re.split(r"\n\s*\n", text)
    current_segment = ""

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        # Si l'ajout du paragraphe dépasse la taille maximale
        if len(current_segment) + len(paragraph) + 2 > max_length:
            if current_segment:
                segments.append(current_segment.strip())

                # Conserver une partie pour le chevauchement si possible
                overlap_text = (
                    current_segment[-overlap:] if len(current_segment) > overlap else ""
                )
                current_segment = overlap_text

        # Ajouter le paragraphe au segment actuel
        if current_segment and not current_segment.endswith("\n\n"):
            current_segment += "\n\n"
        current_segment += paragraph

    # Ajouter le dernier segment s'il n'est pas vide
    if current_segment.strip():
        segments.append(current_segment.strip())

    return segments


def remove_redundant_segments(
    segments: list[str], similarity_threshold: float = 0.85
) -> list[str]:
    """Élimine les segments trop similaires pour éviter les redondances.

    Args:
        segments: Liste de segments à filtrer
        similarity_threshold: Seuil de similarité au-delà duquel les segments sont considérés redondants

    Returns:
        Liste de segments filtrée
    """
    if not segments:
        return []

    filtered_segments = [segments[0]]

    for current in segments[1:]:
        is_redundant = False

        # Méthode simple basée sur le chevauchement de contenu
        for existing in filtered_segments:
            # Calcul d'une métrique simple de similarité basée sur les mots communs
            current_words = set(current.lower().split())
            existing_words = set(existing.lower().split())

            if not current_words or not existing_words:
                continue

            common_words = current_words.intersection(existing_words)
            similarity = len(common_words) / min(
                len(current_words), len(existing_words)
            )

            if similarity > similarity_threshold:
                is_redundant = True
                break

        if not is_redundant:
            filtered_segments.append(current)

    return filtered_segments


def segment_text(text: str, max_length: int = 500) -> list[str]:
    """Segmente le texte en morceaux avec une stratégie améliorée.

    Args:
        text: Texte à segmenter
        max_length: Longueur maximale de chaque segment

    Returns:
        list: Liste des segments de texte
    """
    logger.debug("Entrée segment_text: '%s'", text)
    
    if not text.strip():
        return []

    # Pour les textes courts sans titre, retourner directement le texte
    if len(text) <= max_length and "#" not in text:
        logger.debug("Texte court sans titre, retour direct: '%s'", text)
        return [text]

    # Analyser la structure du document
    document_structure = analyze_markdown_structure(text)
    logger.debug("Structure du document: %s", document_structure)

    # Si le document n'a pas de structure avec des titres, segmenter directement
    if len(document_structure) == 1 and not document_structure[0]["title"]:
        return [text] if len(text) <= max_length else segment_by_semantic_units(text, max_length, overlap=50)

    # Segmenter en respectant les limites naturelles du document
    segments = []
    for section in document_structure:
        # Segmenter chaque section indépendamment
        section_segments = segment_by_semantic_units(
            section["content"],
            max_length,
            overlap=50,  # Chevauchement de 50 caractères entre segments
        )

        # Ajouter le titre de la section uniquement s'il existe
        if section["title"]:
            segments.extend([f"{section['title']}\n\n{s}" for s in section_segments])
        else:
            segments.extend(section_segments)

    # Dédupliquer les segments trop similaires
    final_segments = remove_redundant_segments(segments, similarity_threshold=0.85)
    logger.debug("Segments finaux: %s", final_segments)
    return final_segments
