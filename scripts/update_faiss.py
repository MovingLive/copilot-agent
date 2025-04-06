"""Fichier : update_faiss.py.

Ce script réalise les opérations suivantes :
1. Cloner ou mettre à jour plusieurs dépôts GitHub contenant de la documentation et du code source.
2. Récupérer les discussions GitHub validées via l'API GraphQL.
3. Lire et segmenter les fichiers en morceaux de taille raisonnable.
4. Filtrer les fichiers non pertinents (images, binaires, etc.).
5. Générer des embeddings pour chaque segment à l'aide du modèle "all-MiniLM-L6-v2".
6. Indexer ces embeddings dans un index FAISS.
7. Synchroniser le dossier contenant l'index FAISS vers un bucket AWS S3 ou un répertoire local.
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

# Import le plus tôt possible pour la configuration
from app.core.config import settings

# Autres imports après la configuration
from app.services.embedding_service import EXPECTED_DIMENSION
from app.services.faiss_service import save_faiss_index
from app.utils.document_utils import read_relevant_files
from app.utils.export_utils import export_data, is_local_environment
from app.utils.git_utils import clone_multiple_repos
from app.utils.github_discussions_utils import get_validated_discussions_from_repos
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


def get_repo_urls() -> list[str]:
    """Récupère les URLs des dépôts depuis la variable d'environnement REPO_URLS."""
    repo_urls_env = os.getenv("REPO_URLS", "[]")

    try:
        if repo_urls_env.startswith("[") and repo_urls_env.endswith("]"):
            repo_urls = json.loads(repo_urls_env)
        else:
            repo_urls = [
                url.strip()
                for url in repo_urls_env.replace(";", ",").split(",")
                if url.strip()
            ]
    except json.JSONDecodeError:
        logging.error("Format REPO_URLS invalide")
        return []

    if not repo_urls:
        logging.error("Aucun dépôt spécifié dans REPO_URLS")
        return []

    return repo_urls


def main() -> None:
    """Main function to orchestrate the FAISS index creation and upload process."""
    logging.info("Démarrage du script de mise à jour de l'index FAISS...")

    # Récupérer les URLs des dépôts
    repo_urls = get_repo_urls()

    if not repo_urls:
        logging.error("Aucun dépôt GitHub spécifié dans REPO_URLS ou REPO_URL")
        sys.exit(1)

    # Étape 1 : Cloner ou mettre à jour les dépôts GitHub
    repo_dirs = clone_multiple_repos(repo_urls)

    if not repo_dirs:
        logging.error("Aucun dépôt n'a pu être cloné, arrêt du script")
        sys.exit(1)

    # Étape 2 : Lire et traiter les fichiers pertinents de tous les dépôts
    all_documents = []
    for repo_dir in repo_dirs:
        documents = read_relevant_files(repo_dir)
        all_documents.extend(documents)

    # Étape 3 : Récupérer les discussions GitHub validées
    logging.info("Récupération des discussions GitHub validées...")
    github_discussions = get_validated_discussions_from_repos(repo_urls)
    logging.info(
        "Récupéré %d discussions GitHub validées au total", len(github_discussions)
    )

    # Ajouter les discussions GitHub aux documents à indexer
    all_documents.extend(github_discussions)

    if not all_documents:
        raise ValueError("La liste de documents est vide")

    # Étape 4 : Traiter les documents pour FAISS
    processed_docs = process_files_for_faiss(all_documents, settings.SEGMENT_MAX_LENGTH)

    # Étape 5 : Charger le modèle d'embedding
    model = load_embedding_model()

    # Étape 6 : Générer l'index FAISS et le mapping des métadonnées
    index, metadata_mapping = create_faiss_index(processed_docs, model)

    # Étape 7 : Sauvegarder l'index FAISS et le mapping
    save_faiss_index(index, metadata_mapping, settings.TEMP_FAISS_DIR)

    # Étape 8 : Exporter les données
    is_test_mode = "pytest" in sys.modules
    if not is_local_environment() or is_test_mode:
        logging.info("Exportation des données...")
        export_data(settings.TEMP_FAISS_DIR, settings.S3_BUCKET_PREFIX)
        logging.info("Exportation terminée.")
    else:
        logging.info("Environnement local détecté, pas d'exportation supplémentaire.")


if __name__ == "__main__":
    main()
