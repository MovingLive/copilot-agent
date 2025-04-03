"""Fichier : update_faiss.py.

Ce script réalise les opérations suivantes :
1. Lire le contenu des repositories GitHub via l'API
2. Segmenter les fichiers en morceaux de taille raisonnable
3. Filtrer les fichiers non pertinents (images, binaires, etc.)
4. Générer des embeddings pour chaque segment à l'aide du modèle "all-MiniLM-L6-v2"
5. Indexer ces embeddings dans un index FAISS
6. Synchroniser le dossier contenant l'index FAISS vers un bucket AWS S3 ou un répertoire local
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
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import le plus tôt possible pour la configuration
from app.core.config import settings

# Autres imports après la configuration
from app.services.embedding_service import EXPECTED_DIMENSION
from app.services.faiss_service import save_faiss_index
from app.utils.export_utils import export_data, is_local_environment
from app.utils.git_utils import read_repository_content
from app.utils.vector_db_utils import process_files_for_faiss

# --- Chargement des variables d'environnement ---
load_dotenv()

# --- Configuration du logging ---
logging.basicConfig(level=settings.LOG_LEVEL, format=settings.LOG_FORMAT)


def load_embedding_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Charge le modèle d'embedding SentenceTransformer."""
    logging.info("Chargement du modèle d'embedding '%s'...", model_name)
    return SentenceTransformer(model_name)


def create_faiss_index(
    processed_docs: list, embedding_model: SentenceTransformer
) -> tuple[faiss.IndexIDMap, dict]:
    """Génère les embeddings et crée un index FAISS."""
    if not processed_docs:
        raise ValueError("La liste de documents est vide")

    try:
        texts = [doc["text"] for doc in processed_docs]
        numeric_ids = [doc["numeric_id"] for doc in processed_docs]
        metadata_mapping = {
            str(doc["numeric_id"]): doc["metadata"] for doc in processed_docs
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

    dimension = embeddings.shape[1] if len(embeddings.shape) > 1 else EXPECTED_DIMENSION
    logging.info("Dimension des embeddings: %d", dimension)

    # Création de l'index FAISS
    index = faiss.IndexFlatL2(dimension)
    index_id_map = faiss.IndexIDMap(index)
    np_embeddings = np.array(embeddings).astype("float32")
    np_ids = np.array(numeric_ids, dtype=np.int64)

    if len(np_embeddings.shape) == 1:
        np_embeddings = np_embeddings.reshape(1, -1)
        np_ids = np_ids.reshape(-1)

    index_id_map.add_with_ids(np_embeddings, np_ids)
    logging.info("Index FAISS créé et rempli.")

    return index_id_map, metadata_mapping


def get_repo_list() -> list[str]:
    """Récupère la liste des repositories depuis la variable d'environnement REPO_URLS."""
    repo_urls_env = settings.REPO_URLS

    try:
        if repo_urls_env.startswith("[") and repo_urls_env.endswith("]"):
            repos = json.loads(repo_urls_env)
        else:
            repos = [
                repo.strip()
                for repo in repo_urls_env.replace(";", ",").split(",")
                if repo.strip()
            ]
    except json.JSONDecodeError:
        logging.error("Format REPO_URLS invalide")
        return []

    if not repos:
        logging.error("Aucun repository spécifié dans REPO_URLS")
        return []

    # Valider le format owner/repo
    valid_repos = []
    for repo in repos:
        if "/" in repo and len(repo.split("/")) == 2:
            valid_repos.append(repo)
        else:
            logging.warning(
                "Format de repository invalide '%s'. Utiliser le format 'owner/repo'", repo
            )

    return valid_repos


def main() -> None:
    """Main function to orchestrate the FAISS index creation and upload process."""
    logging.info("Démarrage du script de mise à jour de l'index FAISS...")

    # Récupérer la liste des repositories
    repos = get_repo_list()

    if not repos:
        logging.error("Aucun repository GitHub valide spécifié dans REPO_URLS")
        sys.exit(1)

    # Étape 1 : Lire le contenu des repositories via l'API GitHub
    all_documents = []
    for repo in repos:
        documents = read_repository_content(repo)
        if documents:
            logging.info("Repository %s: %d fichiers lus", repo, len(documents))
            all_documents.extend(documents)
        else:
            logging.warning("Aucun fichier lu depuis le repository %s", repo)

    if not all_documents:
        raise ValueError("Aucun document n'a été lu depuis les repositories")

    # Étape 2 : Traiter les documents pour FAISS
    processed_docs = process_files_for_faiss(all_documents, settings.SEGMENT_MAX_LENGTH)

    # Étape 3 : Charger le modèle d'embedding
    model = load_embedding_model()

    # Étape 4 : Générer l'index FAISS et le mapping des métadonnées
    index, metadata_mapping = create_faiss_index(processed_docs, model)

    # Étape 5 : Sauvegarder l'index FAISS et le mapping
    save_faiss_index(index, metadata_mapping, settings.TEMP_FAISS_DIR)

    # Étape 6 : Exporter les données
    is_test_mode = "pytest" in sys.modules
    if not is_local_environment() or is_test_mode:
        logging.info("Exportation des données...")
        export_data(settings.TEMP_FAISS_DIR, settings.S3_BUCKET_PREFIX)
    else:
        logging.info("Environnement local détecté, pas d'exportation supplémentaire.")

    logging.info("Exportation terminée.")

if __name__ == "__main__":
    main()
