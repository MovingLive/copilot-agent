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

import logging
import os
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

# Ajouter le répertoire parent au chemin pour pouvoir importer les modules de app
sys.path.append(
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from app.utils import (
    clone_or_update_repo,
    export_data,
    process_documents_for_chroma,
    read_markdown_files,
)

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


def main() -> None:
    """
    Fonction principale pour orchestrer la création et l'upload de l'index ChromaDB.
    """
    # Étape 1 : Cloner ou mettre à jour le dépôt GitHub
    repo_dir = clone_or_update_repo(REPO_URL, REPO_DIR)

    # Étape 2 : Lire et traiter les fichiers Markdown
    documents = read_markdown_files(repo_dir)
    processed_docs = process_documents_for_chroma(documents, SEGMENT_MAX_LENGTH)

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

    # Étape 6 : Exporter les données (local ou S3)
    logging.info("Exportation des données...")
    export_data(CHROMA_PERSIST_DIR, S3_BUCKET_PREFIX)
    logging.info("Exportation terminée.")


if __name__ == "__main__":
    main()
