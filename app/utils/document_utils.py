"""Module utilitaire pour la lecture et le filtrage des fichiers.

Ce module contient des fonctions pour déterminer si un fichier est pertinent pour l'indexation
et pour lire différents types de fichiers (code, texte) avec une gestion appropriée des encodages.
"""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Extensions de fichiers à exclure
EXCLUDED_EXTENSIONS: set[str] = {
    # Images
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".bmp",
    ".tiff",
    ".ico",
    ".svg",
    ".webp",
    # Fichiers binaires et compilés
    ".exe",
    ".dll",
    ".so",
    ".dylib",
    ".bin",
    ".obj",
    ".o",
    ".a",
    ".lib",
    ".pyc",
    ".pyo",
    "__pycache__",
    ".class",
    ".jar",
    # Archives
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".iso",
    # Autres formats non textuels
    ".pdf",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    # Grandes bases de données ou fichiers de données
    ".db",
    ".sqlite",
    ".sqlite3",
    ".dat",
    ".sav",
    ".pkl",
    ".pickle",
    # Fichiers spécifiques à certains environnements
    ".env",
    ".env.local",
    ".DS_Store",
    "Thumbs.db",
    ".gitignore",
    ".git",
}

# Extensions de fichiers de code à inclure explicitement
INCLUDED_CODE_EXTENSIONS: set[str] = {
    # Langages de programmation courants
    ".py",
    ".js",
    ".ts",
    ".java",
    ".c",
    ".cpp",
    ".h",
    ".hpp",
    ".cs",
    ".go",
    ".rs",
    ".php",
    ".rb",
    ".pl",
    ".swift",
    ".kt",
    ".scala",
    ".sh",
    ".bash",
    ".ps1",
    # Fichiers web
    ".html",
    ".htm",
    ".css",
    ".scss",
    ".sass",
    ".less",
    ".jsx",
    ".tsx",
    # Scripts et configuration
    ".json",
    ".xml",
    ".yaml",
    ".yml",
    ".toml",
    ".ini",
    ".cfg",
    ".conf",
    # Documentation
    ".md",
    ".markdown",
    ".rst",
    ".txt",
}


def is_file_relevant(file_path: str) -> bool:
    """Détermine si un fichier doit être indexé en fonction de son extension.

    Args:
        file_path: Chemin du fichier à vérifier

    Returns:
        bool: True si le fichier doit être indexé, False sinon
    """
    ext = Path(file_path).suffix.lower()

    # Vérifier d'abord si l'extension est dans la liste des extensions exclues
    if ext in EXCLUDED_EXTENSIONS:
        return False

    # Vérifier si l'extension est dans la liste des extensions incluses
    if ext in INCLUDED_CODE_EXTENSIONS:
        return True

    # Pour les autres extensions, on essaie de déterminer si c'est un fichier texte
    try:
        # Vérifier si le fichier est un fichier texte en essayant de lire quelques lignes
        with open(file_path, encoding="utf-8") as f:
            # Lire quelques lignes pour détecter si c'est du texte
            f.readline()
            return True
    except (OSError, UnicodeDecodeError):
        # Si on ne peut pas lire le fichier comme du texte, on le considère comme non pertinent
        return False


def read_code_file(file_path: str) -> str:
    """Lit le contenu d'un fichier code.

    Args:
        file_path: Chemin du fichier à lire

    Returns:
        str: Contenu du fichier
    """
    try:
        with open(file_path, encoding="utf-8") as f:
            return f.read()
    except UnicodeDecodeError:
        # Essayer avec une autre encodage en cas d'échec
        try:
            with open(file_path, encoding="latin-1") as f:
                return f.read()
        except Exception as e:
            logger.warning("Impossible de lire le fichier %s avec l'encodage latin-1: %s", file_path, e)
            return ""
    except Exception as e:
        logger.warning("Impossible de lire le fichier %s: %s", file_path, e)
        return ""


def read_relevant_files(directory: str) -> list[tuple[str, str]]:
    """Lit tous les fichiers pertinents (Markdown, code, etc.) du dépôt.

    Args:
        directory: Chemin du répertoire contenant les fichiers

    Returns:
        list: Liste de tuples (chemin de fichier, contenu)
    """
    documents = []

    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)

            # Vérifier si le fichier est pertinent
            if is_file_relevant(file_path):
                try:
                    content = read_code_file(file_path)
                    if content.strip():  # Ne pas ajouter les fichiers vides
                        documents.append((file_path, content))
                except Exception as e:
                    logger.warning(
                        "Erreur lors de la lecture du fichier %s: %s", file_path, e
                    )

    logger.info("Nombre de fichiers pertinents lus: %d", len(documents))
    return documents
