#!/usr/bin/env python3
"""
Fichier : update_chroma.py

Ce script réalise les opérations suivantes :
1. Cloner ou mettre à jour un dépôt GitHub contenant la documentation Markdown.
2. Lire et segmenter les fichiers Markdown en morceaux de taille raisonnable.
3. Générer des embeddings pour chaque segment à l'aide du modèle "all-MiniLM-L6-v2".
4. Indexer ces embeddings dans une base ChromaDB et sauvegarder la collection.
5. Synchroniser le dossier contenant la collection ChromaDB vers un bucket AWS S3 ou un répertoire local.

Dépendances :
    - git (disponible en ligne de commande)
    - Python 3
    - poetry add sentence-transformers chromadb boto3
"""

import glob
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import (
    embedding_functions,  # Ajout de l'import pour les fonctions d'embedding
)
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules de app
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from app.utils import export_data

# --- Chargement des variables d'environnement ---
load_dotenv()

# --- Configuration ---

# Environnement (local, dev, prod)
ENV = os.getenv("ENV", "local")

# URL du dépôt GitHub contenant la documentation
REPO_URL = os.getenv(
    "REPO_URL", "https://github.com/votre_utilisateur/votre_repo.git"
)  # Remplacer par l'URL de votre repo
# Dossier local dans lequel le repo sera cloné/actualisé
REPO_DIR = os.getenv("REPO_DIR", "documentation_repo")
# Dossier de persistance de ChromaDB
CHROMA_PERSIST_DIR = "chroma_db"
# Nom de la collection ChromaDB
COLLECTION_NAME = "documentation"

# Paramètres pour la synchronisation S3
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mon-bucket-chroma")
S3_BUCKET_PREFIX = "chroma_db"  # Dossier cible dans le bucket

# Répertoire de sortie local (utilisé si ENV = "local")
LOCAL_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "output"
)

# Paramètre de segmentation des fichiers Markdown
SEGMENT_MAX_LENGTH = 500  # Nombre maximal de caractères par segment

# --- Configuration du logging ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


# --- Fonctions utilitaires ---
def clone_or_update_repo() -> str:
    """
    Clone ou met à jour le dépôt GitHub contenant la documentation.
    Utilise un répertoire temporaire si le répertoire cible n'est pas accessible en écriture.

    Returns:
        str: Chemin du répertoire contenant le dépôt cloné
    """
    # Utiliser le répertoire défini par la variable d'environnement ou créer un répertoire temporaire
    repo_dir = None

    # D'abord, essayer le répertoire configuré
    configured_dir = os.path.abspath(REPO_DIR)

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
                repo_dir = configured_dir
            except (IOError, OSError):
                logging.warning(
                    "Le répertoire %s n'est pas accessible en écriture", configured_dir
                )
        else:
            # Essayer de créer le répertoire
            try:
                os.makedirs(configured_dir, exist_ok=True)
                repo_dir = configured_dir
            except (IOError, OSError) as e:
                logging.warning(
                    "Impossible de créer le répertoire %s: %s", configured_dir, e
                )
    except Exception as e:
        logging.warning("Erreur lors de la vérification des permissions: %s", e)

    # Si on n'a pas pu utiliser le répertoire configuré, utiliser un répertoire temporaire
    if repo_dir is None:
        repo_dir = os.path.join(tempfile.gettempdir(), "repo_clone_" + str(os.getpid()))
        logging.info("Utilisation du répertoire temporaire: %s", repo_dir)
        if not os.path.exists(repo_dir):
            os.makedirs(repo_dir, exist_ok=True)

    # Cloner ou mettre à jour le dépôt
    if not os.path.exists(os.path.join(repo_dir, ".git")):
        logging.info(
            "Clonage du repo depuis %s dans le dossier %s...", REPO_URL, repo_dir
        )
        try:
            subprocess.run(["git", "clone", REPO_URL, repo_dir], check=True)
        except subprocess.CalledProcessError:
            logging.error("Erreur lors du clonage du repo.")
            sys.exit(1)
    else:
        logging.info("Mise à jour du repo dans le dossier %s...", repo_dir)
        try:
            subprocess.run(["git", "-C", repo_dir, "pull"], check=True)
        except subprocess.CalledProcessError:
            logging.error("Erreur lors de la mise à jour du repo.")
            sys.exit(1)

    return repo_dir


def read_markdown_files() -> list[tuple[str, str]]:
    """
    Lit tous les fichiers Markdown du dépôt et retourne une liste de tuples (chemin, contenu).
    """
    markdown_files = glob.glob(os.path.join(REPO_DIR, "**", "*.md"), recursive=True)
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


def segment_text(text: str, max_length: int = SEGMENT_MAX_LENGTH) -> list[str]:
    """
    Segmente le texte en morceaux de longueur maximale max_length.
    Une segmentation simple basée sur les sauts de ligne et la longueur.
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


def process_documents(documents: list[tuple[str, str]]) -> list[dict]:
    """
    Pour chaque document, segmente le contenu et retourne une liste de dictionnaires contenant
    le texte, un identifiant et des métadonnées.
    """
    processed = []
    for file_path, content in documents:
        segments = segment_text(content)
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


def main() -> None:
    """
    Fonction principale pour orchestrer la création et l'upload de l'index ChromaDB.
    """
    # Étape 1 : Cloner ou mettre à jour le dépôt GitHub
    clone_or_update_repo()

    # Étape 2 : Lire et traiter les fichiers Markdown
    documents = read_markdown_files()
    processed_docs = process_documents(documents)

    # Étape 3 : Préparer les données pour ChromaDB
    ids = [doc["id"] for doc in processed_docs]
    texts = [doc["text"] for doc in processed_docs]
    metadatas = [doc["metadata"] for doc in processed_docs]

    # Étape 4 : Configurer et initialiser ChromaDB
    logging.info("Initialisation de ChromaDB...")
    client = chromadb.Client(
        Settings(persist_directory=CHROMA_PERSIST_DIR, anonymized_telemetry=False)
    )

    # Créer une fonction d'embedding compatible avec ChromaDB
    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    # Recréer la collection pour avoir des données fraîches
    try:
        client.delete_collection(COLLECTION_NAME)
    except ValueError:
        pass  # La collection n'existe pas encore, on continue
    collection = client.create_collection(
        name=COLLECTION_NAME, embedding_function=sentence_transformer_ef
    )

    # Étape 5 : Ajouter les documents à la collection ChromaDB
    logging.info("Ajout des documents à la collection ChromaDB...")
    # Ajouter par lots de 200 pour éviter de dépasser les limites de mémoire
    batch_size = 200
    for i in range(0, len(ids), batch_size):
        end = min(i + batch_size, len(ids))
        collection.add(
            ids=ids[i:end], documents=texts[i:end], metadatas=metadatas[i:end]
        )

    # Étape 6 : La persistance est automatique avec ChromaDB quand persist_directory est spécifié
    logging.info("Collection ChromaDB créée et persistée automatiquement...")

    # Étape 7 : Exporter les données (local ou S3)
    logging.info("Exportation des données...")
    export_data(CHROMA_PERSIST_DIR, S3_BUCKET_PREFIX)
    logging.info("Exportation terminée.")


if __name__ == "__main__":
    main()
