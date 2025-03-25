#!/usr/bin/env python3
"""Fichier : update_faiss.py.

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

import json
import logging
import os
import sys

import faiss
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules de app
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from app.core.config import settings
from app.utils import (
    clone_or_update_repo,
    export_data,
    process_documents_for_faiss,
    read_markdown_files,
)

# --- Chargement des variables d'environnement ---
load_dotenv()


# --- Configuration du logging ---
logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)


def create_faiss_index(
    processed_docs: list, embedding_model: SentenceTransformer
) -> tuple[faiss.IndexIDMap, dict]:
    """Génère les embeddings pour chaque segment, crée un index FAISS et renvoie l'index ainsi que
    le mapping des IDs aux métadonnées.

    Args:
        processed_docs: Liste des documents traités avec numeric_id, text et metadata
        embedding_model: Modèle SentenceTransformer pour générer les embeddings

    Returns:
        tuple: (Index FAISS, Mapping des IDs vers les métadonnées)

    Raises:
        ValueError: Si la liste de documents est vide ou invalide
    """
    if not processed_docs:
        raise ValueError("La liste de documents est vide")

    try:
        texts = [doc["text"] for doc in processed_docs]
        numeric_ids = [doc["numeric_id"] for doc in processed_docs]
        metadata_mapping = {
            doc["numeric_id"]: doc["metadata"] for doc in processed_docs
        }
    except (KeyError, TypeError) as e:
        raise ValueError(
            "Format de document invalide. Chaque document doit avoir 'numeric_id', 'text' et 'metadata'"
        ) from e

    # Génération des embeddings
    logging.info("Génération des embeddings...")
    embeddings = embedding_model.encode(texts, show_progress_bar=True)
    embeddings = np.array(embeddings).astype("float32")

    if embeddings.size == 0:
        raise ValueError("Aucun embedding n'a été généré")

    dimension = embeddings.shape[1] if len(embeddings.shape) > 1 else 128

    # Création de l'index FAISS
    index = faiss.IndexFlatL2(dimension)
    # Envelopper avec un IndexIDMap pour associer nos identifiants numériques
    index_id_map = faiss.IndexIDMap(index)
    # Conversion en numpy arrays et ajout à l'index
    np_embeddings = np.array(embeddings).astype("float32")
    np_ids = np.array(numeric_ids, dtype=np.int64)
    # Correction : utilisation de add_with_ids sans noms de paramètres
    index_id_map.add_with_ids(np_embeddings, np_ids)
    logging.info("Index FAISS créé et rempli.")

    return index_id_map, metadata_mapping


def save_faiss_index(
    index: faiss.IndexIDMap, metadata_mapping: dict, directory: str
) -> None:
    """Sauvegarde l'index FAISS et le mapping des IDs dans le répertoire spécifié.

    Args:
        index: Index FAISS à sauvegarder
        metadata_mapping: Mapping des IDs vers les métadonnées
        directory: Répertoire de destination

    Raises:
        PermissionError: Si les permissions sont insuffisantes pour créer/écrire les fichiers
        OSError: Pour les autres erreurs d'E/S
    """
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Convertir les clés en str pour la sérialisation JSON
        str_mapping = {str(k): v for k, v in metadata_mapping.items()}

        index_file_path = os.path.join(directory, settings.FAISS_INDEX_FILE)
        mapping_file_path = os.path.join(directory, settings.FAISS_METADATA_FILE)

        faiss.write_index(index, index_file_path)
        with open(mapping_file_path, "w", encoding="utf-8") as f:
            json.dump(str_mapping, f, ensure_ascii=False, indent=2)
        logging.info("Index FAISS et mapping sauvegardés localement.")
    except (PermissionError, OSError) as e:
        logging.error("Erreur lors de la sauvegarde de l'index : %s", e)
        raise


def main() -> None:
    """Main function to orchestrate the FAISS index creation and upload process."""
    logging.info("Démarrage du script de mise à jour de l'index FAISS...")

    # Étape 1 : Cloner ou mettre à jour le dépôt GitHub
    repo_dir = clone_or_update_repo(settings.REPO_URL, settings.REPO_DIR)

    # Étape 2 : Lire et traiter les fichiers Markdown
    documents = read_markdown_files(repo_dir)
    processed_docs = process_documents_for_faiss(documents, settings.SEGMENT_MAX_LENGTH)

    # Étape 3 : Charger le modèle d'embedding
    logging.info("Chargement du modèle d'embedding 'all-MiniLM-L6-v2'...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Étape 4 : Générer l'index FAISS et le mapping des métadonnées
    index, metadata_mapping = create_faiss_index(processed_docs, model)

    # Étape 5 : Sauvegarder l'index FAISS et le mapping localement
    save_faiss_index(index, metadata_mapping, settings.TEMP_FAISS_DIR)

    # Étape 6 : Exporter les données (local ou S3)
    logging.info("Exportation des données...")
    export_data(settings.TEMP_FAISS_DIR, settings.S3_BUCKET_PREFIX)
    logging.info("Exportation terminée.")


if __name__ == "__main__":
    main()
