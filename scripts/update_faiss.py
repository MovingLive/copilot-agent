#!/usr/bin/env python3
"""
Fichier : update_faiss.py

Ce script réalise les opérations suivantes :
1. Cloner ou mettre à jour un dépôt GitHub contenant la documentation Markdown.
2. Lire et segmenter les fichiers Markdown en morceaux de taille raisonnable.
3. Générer des embeddings pour chaque segment à l'aide du modèle "all-MiniLM-L6-v2" de SentenceTransformers.
4. Indexer ces embeddings dans un index FAISS et sauvegarder l'index ainsi qu'un mapping des IDs aux métadonnées.
5. Synchroniser le dossier contenant l'index FAISS vers un bucket AWS S3 ou un répertoire local.

Dépendances :
    - git (disponible en ligne de commande)
    - Python 3
    - poetry add sentence-transformers faiss-cpu boto3
"""

import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from app.utils import export_data

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules de app
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# --- Chargement des variables d'environnement ---
load_dotenv()


# --- Configuration ---

# Environnement (local, dev, prod)
ENV = os.getenv("ENV", "local")

# URL du dépôt GitHub contenant la documentation
REPO_URL = os.getenv("REPO_URL", "https://github.com/votre_utilisateur/votre_repo.git")
# Dossier local dans lequel le repo sera cloné/actualisé
REPO_DIR = os.getenv("REPO_DIR", "documentation_repo")
# Dossier de persistance de l'index FAISS
FAISS_PERSIST_DIR = os.path.join(tempfile.gettempdir(), "faiss_index")
# Nom du fichier d'index FAISS
FAISS_INDEX_FILE = "faiss.index"
# Nom du fichier de mapping des IDs (pour retrouver les métadonnées)
ID_MAPPING_FILE = "id_mapping.json"

# Paramètres pour la synchronisation S3
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mon-bucket-faiss")
S3_BUCKET_PREFIX = "faiss_index"  # Dossier cible dans le bucket

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
    le texte, un identifiant numérique et des métadonnées.
    """
    processed = []
    current_id = 0
    for file_path, content in documents:
        segments = segment_text(content)
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


def create_faiss_index(
    processed_docs: list, embedding_model: SentenceTransformer
) -> tuple[faiss.IndexIDMap, dict]:
    """
    Génère les embeddings pour chaque segment, crée un index FAISS et renvoie l'index ainsi que
    le mapping des IDs aux métadonnées.
    """
    texts = [doc["text"] for doc in processed_docs]
    numeric_ids = [doc["numeric_id"] for doc in processed_docs]
    metadata_mapping = {doc["numeric_id"]: doc["metadata"] for doc in processed_docs}

    # Génération des embeddings
    logging.info("Génération des embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")
    dimension = embeddings.shape[1]
    logging.info("Dimension des embeddings : %d", dimension)

    # Création de l'index FAISS
    index = faiss.IndexFlatL2(dimension)
    # Envelopper avec un IndexIDMap pour associer nos identifiants numériques
    index_id_map = faiss.IndexIDMap(index)
    index_id_map.add_with_ids(x=embeddings, xids=np.array(numeric_ids, dtype=np.int64))
    logging.info("Index FAISS créé et rempli.")

    return index_id_map, metadata_mapping


def save_faiss_index(
    index: faiss.IndexIDMap, metadata_mapping: dict, directory: str
) -> None:
    """
    Sauvegarde l'index FAISS et le mapping des IDs dans le répertoire spécifié.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
    index_file_path = os.path.join(directory, FAISS_INDEX_FILE)
    mapping_file_path = os.path.join(directory, ID_MAPPING_FILE)

    faiss.write_index(index, index_file_path)
    with open(mapping_file_path, "w", encoding="utf-8") as f:
        json.dump(metadata_mapping, f, ensure_ascii=False, indent=2)
    logging.info("Index FAISS et mapping sauvegardés localement.")


def main() -> None:
    """Main function to orchestrate the FAISS index creation and upload process."""

    logging.info("Démarrage du script de mise à jour de l'index FAISS...")

    # Étape 1 : Cloner ou mettre à jour le dépôt GitHub
    repo_dir = clone_or_update_repo()

    # Utiliser le répertoire retourné au lieu de la constante REPO_DIR
    global REPO_DIR
    REPO_DIR = repo_dir

    # Étape 2 : Lire et traiter les fichiers Markdown
    documents = read_markdown_files()
    processed_docs = process_documents(documents)

    # Étape 3 : Charger le modèle d'embedding
    logging.info("Chargement du modèle d'embedding 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Étape 4 : Générer l'index FAISS et le mapping des métadonnées
    index, metadata_mapping = create_faiss_index(processed_docs, model)

    # Étape 5 : Sauvegarder l'index FAISS et le mapping localement
    save_faiss_index(index, metadata_mapping, FAISS_PERSIST_DIR)

    # Étape 6 : Exporter les données (local ou S3)
    logging.info("Exportation des données...")
    export_data(FAISS_PERSIST_DIR, S3_BUCKET_PREFIX)
    logging.info("Exportation terminée.")


if __name__ == "__main__":
    main()
