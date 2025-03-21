#!/usr/bin/env python3
"""
Fichier : update_faiss.py

Ce script réalise les opérations suivantes :
1. Cloner ou mettre à jour un dépôt GitHub contenant la documentation Markdown.
2. Lire et segmenter les fichiers Markdown en morceaux de taille raisonnable.
3. Générer des embeddings pour chaque segment à l'aide du modèle "all-MiniLM-L6-v2" de SentenceTransformers.
4. Indexer ces embeddings dans un index FAISS et sauvegarder l'index ainsi qu'un mapping des IDs aux métadonnées.
5. Synchroniser le dossier contenant l'index FAISS vers un bucket AWS S3.

Dépendances :
    - git (disponible en ligne de commande)
    - Python 3
    - pip install sentence-transformers faiss-cpu boto3
"""

import os
import sys
import glob
import json
import logging
import subprocess
from pathlib import Path

import numpy as np
import boto3
from sentence_transformers import SentenceTransformer
import faiss

# --- Configuration ---

# URL du dépôt GitHub contenant la documentation
REPO_URL = "https://github.com/votre_utilisateur/votre_repo.git"  # Remplacer par l'URL de votre repo
# Dossier local dans lequel le repo sera cloné/actualisé
REPO_DIR = "documentation_repo"
# Dossier de persistance de l'index FAISS
FAISS_PERSIST_DIR = "faiss_index"
# Nom du fichier d'index FAISS
FAISS_INDEX_FILE = "faiss.index"
# Nom du fichier de mapping des IDs (pour retrouver les métadonnées)
ID_MAPPING_FILE = "id_mapping.json"

# Paramètres pour la synchronisation S3
S3_BUCKET_NAME = "nom-de-votre-bucket-s3"  # Remplacer par le nom de votre bucket
S3_BUCKET_PREFIX = "faiss_index"  # Dossier cible dans le bucket

# Paramètre de segmentation des fichiers Markdown
SEGMENT_MAX_LENGTH = 500  # Nombre maximal de caractères par segment

# --- Configuration du logging ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


# --- Fonctions utilitaires ---

def clone_or_update_repo():
    """
    Clone ou met à jour le dépôt GitHub contenant la documentation.
    """
    if not os.path.exists(REPO_DIR):
        logging.info(f"Clonage du repo depuis {REPO_URL} dans le dossier {REPO_DIR}...")
        try:
            subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True)
        except subprocess.CalledProcessError:
            logging.error("Erreur lors du clonage du repo.")
            sys.exit(1)
    else:
        logging.info(f"Mise à jour du repo dans le dossier {REPO_DIR}...")
        try:
            subprocess.run(["git", "-C", REPO_DIR, "pull"], check=True)
        except subprocess.CalledProcessError:
            logging.error("Erreur lors de la mise à jour du repo.")
            sys.exit(1)


def read_markdown_files():
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
        except Exception as e:
            logging.warning(f"Impossible de lire le fichier {file_path}: {e}")
    logging.info(f"Nombre de fichiers Markdown lus: {len(documents)}")
    return documents


def segment_text(text, max_length=SEGMENT_MAX_LENGTH):
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
                segments.append(para[i:i+max_length])
        else:
            segments.append(para)
    return segments


def process_documents(documents):
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
                    "segment_index": idx
                }
            }
            processed.append(entry)
            current_id += 1
    logging.info(f"Nombre total de segments générés: {len(processed)}")
    return processed


def create_faiss_index(processed_docs, embedding_model):
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
    logging.info(f"Dimension des embeddings : {dimension}")

    # Création de l'index FAISS
    index = faiss.IndexFlatL2(dimension)
    # Envelopper avec un IndexIDMap pour associer nos identifiants numériques
    index_id_map = faiss.IndexIDMap(index)
    index_id_map.add_with_ids(embeddings, np.array(numeric_ids, dtype=np.int64))
    logging.info("Index FAISS créé et rempli.")

    return index_id_map, metadata_mapping


def save_faiss_index(index, metadata_mapping, directory):
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


def upload_directory_to_s3(directory, bucket_name, prefix):
    """
    Upload l'intégralité des fichiers du répertoire 'directory' vers le bucket S3 sous le préfixe 'prefix'.
    """
    s3 = boto3.client('s3')
    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory)
            s3_key = os.path.join(prefix, relative_path).replace("\\", "/")
            logging.info(f"Téléversement de {full_path} vers s3://{bucket_name}/{s3_key}...")
            try:
                s3.upload_file(full_path, bucket_name, s3_key)
            except Exception as e:
                logging.error(f"Erreur lors du téléversement de {full_path}: {e}")


def main():
    # Étape 1 : Cloner ou mettre à jour le dépôt GitHub
    clone_or_update_repo()

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

    # Étape 6 : Synchroniser le dossier FAISS avec le bucket S3
    logging.info("Début de la synchronisation du dossier FAISS vers S3...")
    upload_directory_to_s3(FAISS_PERSIST_DIR, S3_BUCKET_NAME, S3_BUCKET_PREFIX)
    logging.info("Synchronisation terminée.")


if __name__ == "__main__":
    main()
