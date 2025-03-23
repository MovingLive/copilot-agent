import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List

import boto3
import faiss
import httpx
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from sentence_transformers import SentenceTransformer

from app.utils.export_utils import (
    LOCAL_OUTPUT_DIR,
    is_local_environment,
)

# --- Chargement des variables d'environnement ---
load_dotenv()

# --- Configuration ---
ENV = os.getenv("ENV", "local")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mon-bucket-faiss")

FAISS_METADATA_FILE = os.getenv("FAISS_METADATA_FILE", "metadata.json")
FAISS_INDEX_FILE = os.getenv("FAISS_INDEX_FILE", "index.faiss")

AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")
COPILOT_API_URL = os.getenv(
    "COPILOT_API_URL", "https://api.githubcopilot.com/chat/completions"
)
COPILOT_TOKEN = os.getenv("COPILOT_TOKEN", "")

# --- Variables globales ---
FAISS_INDEX = None
document_store = []
# Initialisation du modèle d'embedding à None (sera chargé au démarrage)
EMBEDDING_MODEL = None

# --- Logger ---
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("copilot_api")


# --- Événement de démarrage ---
@asynccontextmanager
async def lifespan(_: FastAPI):
    """
    Gère le cycle de vie de l'application FastAPI.
    """
    logger.info("Démarrage de l'application et chargement de l'index FAISS...")
    # Charger le modèle d'embedding
    global EMBEDDING_MODEL
    try:
        logger.info("Chargement du modèle d'embedding...")
        EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Modèle d'embedding chargé avec succès.")
    except Exception as e:
        logger.error("Erreur lors du chargement du modèle d'embedding: %s", e)

    load_faiss_index()
    # Lancer la mise à jour périodique de l'index dans un thread de fond
    threading.Thread(target=update_faiss_index_periodically, daemon=True).start()
    logger.info("Service de mise à jour de l'index lancé en arrière-plan.")

    yield
    logger.info("Arrêt de l'application. Nettoyage des ressources...")


# --- Instance FastAPI ---
app = FastAPI(lifespan=lifespan)


# --- Fonctions utilitaires ---
def get_local_faiss_path() -> str:
    """
    Détermine le chemin local pour l'index FAISS et le crée si nécessaire.

    Returns:
        str: Chemin vers le fichier d'index FAISS local

    Règle appliquée: Modularisation - Séparation des responsabilités
    """
    if is_local_environment():
        # Charger depuis le répertoire local
        local_faiss_path = os.path.join(LOCAL_OUTPUT_DIR, FAISS_INDEX_FILE)
        if not os.path.exists(local_faiss_path):
            logger.warning(
                "Index FAISS introuvable dans le répertoire local, création d'un index vide."
            )
            local_faiss_path = f"/tmp/{FAISS_INDEX_FILE}"
            # Créer un index vide si aucun n'existe
            # Utiliser la même dimension que le modèle d'embedding
            empty_index = faiss.IndexFlatL2(384)  # Dimension du modèle all-MiniLM-L6-v2

            faiss.write_index(empty_index, local_faiss_path)
    else:
        # Télécharger l'index depuis S3 vers un chemin temporaire
        local_faiss_path = f"/tmp/{FAISS_INDEX_FILE}"
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        s3_client.download_file(S3_BUCKET_NAME, FAISS_INDEX_FILE, local_faiss_path)
        logger.info("Index FAISS téléchargé depuis S3.")

    return local_faiss_path


def load_and_validate_faiss_index(local_faiss_path: str) -> faiss.Index:
    """
    Charge l'index FAISS depuis un fichier local et valide son intégrité.

    Args:
        local_faiss_path (str): Chemin vers le fichier d'index FAISS

    Returns:
        faiss.Index: L'index FAISS chargé

    Règle appliquée: Error Handling - Validation et diagnostique de l'index
    """
    index = faiss.read_index(local_faiss_path)
    logger.info("Index FAISS chargé en mémoire.")

    # Règle appliquée: Python Usage - Utilisation du lazy % formatting dans les logging
    logger.info(
        "Index FAISS chargé: type=%s, dimension=%d, nombre d'éléments=%d",
        type(index),
        index.d,
        index.ntotal,
    )
    logger.info(
        "Est-ce que l'index est entraîné? %s",
        hasattr(index, "is_trained") and index.is_trained,
    )

    if index.ntotal == 0:
        logger.warning(
            "ATTENTION: L'index FAISS est vide (ntotal=0). Aucun résultat de recherche ne sera retourné."
        )

    return index


def get_metadata_path() -> str:
    """
    Détermine le chemin d'accès aux métadonnées associées à l'index FAISS.

    Returns:
        str: Chemin vers le fichier de métadonnées

    Raises:
        FileNotFoundError: Si le fichier de métadonnées n'est pas trouvé en local
    """
    local_metadata_path = f"/tmp/{FAISS_METADATA_FILE}"
    if is_local_environment():
        metadata_path = os.path.join(LOCAL_OUTPUT_DIR, FAISS_METADATA_FILE)
        if os.path.exists(metadata_path):
            local_metadata_path = metadata_path
        else:
            raise FileNotFoundError(
                "Fichier de métadonnées introuvable dans le répertoire local"
            )
    else:
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        s3_client.download_file(
            S3_BUCKET_NAME, FAISS_METADATA_FILE, local_metadata_path
        )

    return local_metadata_path


def load_and_validate_document_store(metadata_path: str) -> dict:
    """
    Charge et valide le mapping des documents (metadata) associé à l'index FAISS.

    Args:
        metadata_path (str): Chemin vers le fichier de métadonnées

    Returns:
        dict: Le mapping des documents

    Règle appliquée: Error Handling - Validation du document store
    """
    with open(metadata_path, "r", encoding="utf-8") as f:
        doc_store = json.load(f)

    logger.info("Mapping des documents chargé avec %d entrées.", len(doc_store))

    # Analyse du contenu du document_store pour diagnostic
    if isinstance(doc_store, dict):
        sample_keys = list(doc_store.keys())[:5]
        logger.info("Échantillon de clés dans document_store: %s", sample_keys)
        if sample_keys:
            sample_doc = doc_store[sample_keys[0]]
            logger.info(
                "Structure d'un document type: %s",
                json.dumps(sample_doc, indent=2)[:200] + "...",
            )
    elif isinstance(doc_store, list):
        logger.info("document_store est une liste de %d éléments", len(doc_store))
        if doc_store:
            logger.info(
                "Structure du premier élément: %s",
                json.dumps(doc_store[0], indent=2)[:200] + "...",
            )

    return doc_store


def load_faiss_index() -> None:
    """
    Télécharge et charge l'index FAISS depuis AWS S3 ou le répertoire local.
    Optionnellement, charge aussi un mapping vers les documents.

    Règle appliquée: Modularisation - Orchestration des sous-fonctions
    """
    global FAISS_INDEX, document_store
    try:
        # Étape 1: Obtenir le chemin du fichier FAISS et le créer si nécessaire
        local_faiss_path = get_local_faiss_path()

        # Étape 2: Charger et valider l'index FAISS
        FAISS_INDEX = load_and_validate_faiss_index(local_faiss_path)

        # Étape 3: Charger le mapping de documents associé (optionnel)
        try:
            # Étape 3.1: Obtenir le chemin du fichier de métadonnées
            metadata_path = get_metadata_path()

            # Étape 3.2: Charger et valider le document store
            document_store = load_and_validate_document_store(metadata_path)

        except Exception as e:
            logger.warning(
                "Mapping des documents introuvable, utilisation d'une liste vide: %s", e
            )
            document_store = []
    except Exception as e:
        logger.error("Erreur lors du chargement de l'index FAISS : %s", e)
        FAISS_INDEX = None
        document_store = []


def update_faiss_index_periodically():
    """
    Tâche en arrière-plan qui met à jour l'index FAISS toutes les 24 heures.
    Ici, on recharge simplement l'index depuis S3.
    La logique peut être étendue pour re-calculer les embeddings à partir de nouvelles données.
    """
    while True:
        logger.info("Démarrage de la mise à jour de l'index FAISS.")
        load_faiss_index()
        logger.info(
            "Mise à jour de l'index FAISS terminée. Attente de 24h avant la prochaine mise à jour."
        )
        time.sleep(24 * 3600)


def embed_text(text: str) -> List[float]:
    """
    Génère un embedding vectoriel à partir d'un texte en utilisant un modèle pré-entraîné.

    Args:
        text (str): Le texte à transformer en embedding

    Returns:
        List[float]: Le vecteur d'embedding normalisé représentant le texte
    """
    global EMBEDDING_MODEL

    try:
        # Vérifier si le modèle est chargé
        if EMBEDDING_MODEL is None:
            logger.warning("Modèle d'embedding non chargé, tentative de chargement...")
            EMBEDDING_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

        # Générer l'embedding
        embedding = EMBEDDING_MODEL.encode(text, normalize_embeddings=True)

        # Normalisation min-max pour garantir des valeurs entre 0 et 1
        if embedding.max() != embedding.min():  # Éviter la division par zéro
            embedding = (embedding - embedding.min()) / (
                embedding.max() - embedding.min()
            )

        # Log des caractéristiques de l'embedding pour diagnostic
        logger.info(
            "Embedding généré avec succès: min=%f, max=%f, norme=%f",
            np.min(embedding),
            np.max(embedding),
            np.linalg.norm(embedding),
        )

        return embedding.tolist()

    except Exception as e:
        logger.error("Erreur lors de la génération de l'embedding: %s", e)
        # Fallback en cas d'erreur
        dim = 384  # Dimension standard pour le modèle all-MiniLM-L6-v2
        rng = np.random.RandomState(abs(hash(text)) % (2**32))
        vector = rng.rand(dim).astype(
            "float32"
        )  # Utiliser rand pour générer des valeurs entre 0 et 1
        logger.warning("Utilisation d'un vecteur aléatoire normalisé en fallback")
        return vector.tolist()


def generate_query_vector(query: str) -> np.ndarray:
    """
    Génère un vecteur d'embedding à partir d'une requête textuelle.

    Args:
        query (str): La requête textuelle

    Returns:
        np.ndarray: Le vecteur d'embedding normalisé

    Raises:
        ValueError: Si la requête est vide ou si l'embedding échoue
    """
    if not query:
        raise ValueError("Query cannot be empty")

    logger.info("Génération de l'embedding pour la requête...")
    query_vector = np.array([embed_text(query)]).astype("float32")
    logger.info(
        "Embedding généré: shape=%s, min=%.4f, max=%.4f",
        query_vector.shape,
        query_vector.min(),
        query_vector.max(),
    )

    if query_vector.size == 0:
        raise ValueError("Failed to generate embedding for query")

    return query_vector


def ensure_compatible_dimensions(query_vector: np.ndarray, faiss_index) -> np.ndarray:
    """
    Vérifie la compatibilité des dimensions entre le vecteur de requête et l'index FAISS.
    Ajuste le vecteur de requête si nécessaire.

    Args:
        query_vector (np.ndarray): Le vecteur de requête
        faiss_index: L'index FAISS

    Returns:
        np.ndarray: Le vecteur de requête ajusté si nécessaire
    """
    if query_vector.shape[1] != faiss_index.d:
        logger.error(
            "Incompatibilité de dimensions: vecteur de requête (%d) != index FAISS (%d)",
            query_vector.shape[1],
            faiss_index.d,
        )
        # Ajuster dynamiquement la dimension du vecteur de requête
        new_query_vector = np.zeros((1, faiss_index.d), dtype="float32")
        min_dim = min(query_vector.shape[1], faiss_index.d)
        new_query_vector[0, :min_dim] = query_vector[0, :min_dim]
        query_vector = new_query_vector
        logger.info("Vecteur de requête redimensionné à la dimension %d", faiss_index.d)

    return query_vector


def search_faiss_index(query_vector: np.ndarray, faiss_index, k: int) -> tuple:
    """
    Recherche dans l'index FAISS les k plus proches voisins.

    Args:
        query_vector (np.ndarray): Le vecteur de requête
        faiss_index: L'index FAISS
        k (int): Nombre de résultats à retourner

    Returns:
        tuple: (distances, indices) tuple des distances et indices des documents similaires

    Raises:
        HTTPException: En cas d'erreur lors de la recherche
    """
    try:
        logger.info("Exécution de FAISS_INDEX.search avec k=%d", k)
        distances, indices = faiss_index.search(query_vector, k)

        # Analyse statistique des distances pour diagnostic
        logger.info(
            "Statistiques des distances: min=%.4f, max=%.4f, moyenne=%.4f",
            distances.min(),
            distances.max(),
            distances.mean(),
        )
        if distances.mean() > 100.0:
            logger.warning("Distances vectorielles très élevées, ajustement de k")
            # Récupérer plus de documents pour compenser la faible qualité
            distances, indices = faiss_index.search(query_vector, k*2)

        # Règle appliquée: Python Usage - Utilisation du lazy % formatting dans les logging functions
        logger.info("Indices trouvés: %s", indices[0])
        logger.info("Distances associées: %s", distances[0])
        if np.all(indices[0] == -1):
            logger.warning(
                "ATTENTION: Tous les indices retournés sont -1, ce qui indique qu'aucun résultat n'a été trouvé"
            )

        return distances, indices

    except AssertionError as ae:
        logger.error(
            "AssertionError dans FAISS_INDEX.search: %s",
            str(ae) if str(ae) else "Assertion vide",
        )
        logger.error("Détails de l'index FAISS: %s", str(faiss_index))
        logger.error("Est-ce que l'index est vide? %s", faiss_index.ntotal == 0)
        logger.error(
            "Vecteur de requête (premiers éléments): %s", query_vector.flatten()[:5]
        )
        # Règle appliquée: Python Usage - Explicitly re-raising avec from
        raise HTTPException(
            status_code=500, detail="Erreur d'assertion lors de la recherche FAISS"
        ) from ae


def extract_documents_from_indices(
    indices: np.ndarray, distances: np.ndarray, doc_store: dict
) -> List[dict]:
    """
    Extrait les documents correspondant aux indices trouvés.

    Args:
        indices (np.ndarray): Les indices des documents similaires
        distances (np.ndarray): Les distances correspondantes
        doc_store (dict): Le stockage des documents

    Returns:
        List[dict]: Liste des documents extraits avec leur contenu
    """
    results = []
    logger.info(
        "Tentative de récupération des documents à partir des indices: %s",
        indices[0],
    )

    for i, idx in enumerate(indices[0]):
        if idx >= 0:
            doc_key = str(idx)
            logger.info(
                "Recherche du document avec clé=%s, distance=%.4f",
                doc_key,
                distances[0][i],
            )

            if doc_key in doc_store:
                doc = doc_store[doc_key]
                logger.info(
                    "Document trouvé pour l'indice %d. Clés disponibles: %s",
                    idx,
                    list(doc.keys()) if isinstance(doc, dict) else "Non dict",
                )

                # Extraire le contenu complet avec vérification de toutes les structures possibles
                content = None

                # Vérification directe de la clé content
                if isinstance(doc, dict) and "content" in doc:
                    content = doc["content"]
                    logger.info(
                        "Contenu trouvé directement (taille: %d)",
                        len(content) if content else 0,
                    )

                # Log du document complet pour debug
                if isinstance(doc, dict):
                    logger.debug(
                        "Structure complète du document: %s",
                        json.dumps(doc, indent=2),
                    )
                else:
                    logger.debug("Document non-dictionnaire: %s", str(doc)[:100])

                # Ajouter aux résultats si contenu trouvé
                if content:
                    # Log du contenu complet pour vérification (en mode debug)
                    logger.debug(
                        "Contenu complet: %s",
                        content[:200] + "..." if len(content) > 200 else content,
                    )
                    results.append({"content": content})
                    logger.info(
                        "Document ajouté aux résultats (longueur: %d caractères)",
                        len(content),
                    )
                else:
                    logger.warning(
                        "Aucun contenu trouvé dans le document: %s",
                        str(doc)[:100] + "..." if len(str(doc)) > 100 else str(doc),
                    )
            else:
                logger.warning("Indice %d non trouvé dans document_store", idx)

    logger.info("Nombre de résultats retournés: %d", len(results))

    if not results:
        logger.warning("No matching documents found")

    return results


def retrieve_similar_documents(query: str, k: int = 5) -> List[dict]:
    """
    Recherche dans l'index FAISS les k documents les plus proches de la requête.
    Retourne une liste de dictionnaires représentant les documents.
    """
    try:
        logger.info(
            "Début de retrieve_similar_documents avec query='%s...' et k=%d",
            query[:50],
            k,
        )

        if FAISS_INDEX is None:
            logger.error("Index FAISS non chargé.")
            return []

        # Étape 1: Générer le vecteur de requête
        query_vector = generate_query_vector(query)

        # Étape 2: Assurer la compatibilité des dimensions
        query_vector = ensure_compatible_dimensions(query_vector, FAISS_INDEX)

        # Étape 3: Rechercher dans l'index FAISS
        distances, indices = search_faiss_index(query_vector, FAISS_INDEX, k)

        # Étape 4: Extraire les documents à partir des indices
        results = extract_documents_from_indices(indices, distances, document_store)

        return results

    except ValueError as ve:
        logger.error("Validation error in retrieve_similar_documents: %s", ve)
        # Règle appliquée: Python Usage - Explicitly re-raising avec from
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        logger.error("Error in retrieve_similar_documents: %s", e)
        logger.error("Type d'exception: %s", type(e).__name__)
        # Règle appliquée: Error Handling - Logging approprié pour le débogage
        raise HTTPException(
            status_code=500,
            detail="An error occurred while searching similar documents",
        ) from e


def call_copilot_llm(question: str, context_text: str, auth_token: str) -> str:
    """
    Appelle l'API GitHub Copilot LLM en fournissant une question et le contexte récupéré.
    Retourne la réponse générée.

    L'API Copilot nécessite un token d'authentification GitHub au format Bearer.
    """
    # Règle appliquée: Sécurité - Format d'authentification correct selon la documentation
    headers = {
        "authorization": f"Bearer {auth_token}",  # En minuscules et au format Bearer comme dans l'exemple
        "content-type": "application/json",
    }

    # Format compatible avec l'API OpenAI comme indiqué dans la documentation
    payload = {
        "messages": [
            {
                "role": "system",
                "content": "Tu es un assistant spécialisé qui doit répondre en se basant PRINCIPALEMENT sur les informations fournies dans le contexte. Si le contexte contient une réponse à la question, utilise-la en priorité."
            },
            {"role": "system", "content": f"Contexte:\n{context_text}"},
            {"role": "user", "content": question},
        ],
    }

    try:
        logger.info("Envoi de la requête à l'API Copilot LLM")
        response = requests.post(
            COPILOT_API_URL, json=payload, headers=headers, timeout=30
        )

        # Log détaillé en cas d'erreur pour faciliter le débogage
        if not response.ok:
            logger.error(
                "Échec de l'appel API Copilot (%d): %s",
                response.status_code,
                response.text,
            )

        response.raise_for_status()
        data = response.json()

        if (
            "choices" not in data
            or not data["choices"]
            or "message" not in data["choices"][0]
        ):
            logger.error("Format de réponse invalide: %s", data)
            raise ValueError("Format de réponse inattendu de l'API Copilot")

        answer = data["choices"][0]["message"]["content"]
        logger.info("Réponse reçue de l'API Copilot LLM")
        return answer

    except requests.exceptions.HTTPError as http_err:
        if http_err.response.status_code == 400:
            logger.error("Erreur 400 Bad Request: %s", http_err.response.text)
            raise HTTPException(
                status_code=400,
                detail="Format de requête incorrect pour l'API Copilot. Vérifiez la structure de la payload.",
            ) from http_err
        elif http_err.response.status_code == 401:
            logger.error("Erreur 401 Unauthorized: %s", http_err.response.text)
            raise HTTPException(
                status_code=401,
                detail="Token d'authentification Copilot invalide ou expiré.",
            ) from http_err
        else:
            logger.error(
                "Erreur HTTP lors de l'appel à l'API Copilot LLM : %s", http_err
            )
            raise HTTPException(
                status_code=http_err.response.status_code,
                detail=f"Erreur lors de l'appel au service Copilot LLM: {http_err.response.text}",
            ) from http_err
    except Exception as e:
        logger.error("Erreur lors de l'appel à l'API Copilot LLM : %s", e)
        raise HTTPException(
            status_code=500, detail="Erreur lors de l'appel au service Copilot LLM."
        ) from e


async def handle_copilot_query(request: Request) -> StreamingResponse:
    """
    Handler pour traiter les requêtes à l'API Copilot.
    """
    auth_token = request.headers.get("x-github-token")
    if not auth_token:
        logger.error("Token d'authentification manquant dans l'en-tête.")
        raise HTTPException(status_code=401, detail="Missing authentication token")

    data = await request.json()
    messages = data.get("messages", [])
    message = messages[-1].get("content", "")
    logger.info("Message reçu: %s", message)

    additional_context = data.get("copilot_references", "")

    # Récupérer des documents similaires à la question
    docs = retrieve_similar_documents(message + " " + additional_context, k=5)
    # Améliorer le formatage du contexte pour le LLM
    context_sections = []
    for doc in docs:
        content = doc.get("content", "")
        if content and len(content) > 10:  # Éviter les segments trop courts
            context_sections.append(content)

    # Structurer le contexte pour une meilleure utilisation par le LLM
    context_text = "\n\n".join(context_sections)

    # Instruction explicite pour utiliser le contexte
    formatted_context = f"""
        CONTEXTE PERTINENT:
        {context_text}

        INSTRUCTIONS:
        - Utilise le contexte ci-dessus pour répondre à la question de l'utilisateur.
        - Si le contexte contient une réponse directe à la question, base ta réponse dessus.
        - Structure ta réponse de façon claire et précise.
        """

    logger.info("Contexte récupéré via FAISS.")

    # Appeler l'API Copilot LLM
    answer = call_copilot_llm(message, formatted_context, auth_token)

    # Récupérer les informations de l'utilisateur
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(
                "https://api.github.com/user",
                headers={"Authorization": f"Bearer {auth_token}"},
            )
            response.raise_for_status()
            user_data = response.json()
            user_login = user_data.get("login")
            logger.info("Utilisateur authentifié: %s", user_login)
        except Exception as e:
            logger.error("Erreur lors de l'authentification GitHub: %s", e)
            raise HTTPException(status_code=401, detail="Invalid GitHub token") from e

    # Ajouter les messages système
    messages.insert(
        0,
        {
            "role": "system",
            "content": "Tu es un assistant spéclialisé dans les GitHub actions et les workflows. Tu es capable de répondre à des questions sur les GitHub actions, les workflows, et d'autres sujets techniques.",
        },
    )
    messages.insert(
        0,
        {
            "role": "system",
            "content": f"Commence chaque réponse par le nom de l'utilisateur, qui est @{user_login}",
        },
    )

    if answer:
        messages.insert(
            0,
            {
                "role": "system",
                "content": f"Utilise les informations suivantes pour enrichir ta réponse: {answer}",
            },
        )
    messages.insert(
        0,
        {
            "role": "system",
            "content": "Pour chaque réponse, termine par une section 'Le savais-tu?' qui apporte une information technique pertinente et intéressante en lien avec la réponse donnée. Cette section doit être formatée ainsi : '\n\nLe savais-tu ? [Information technique]'",
        },
    )

    async def response_generator():
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST",
                COPILOT_API_URL,
                headers={
                    "authorization": f"Bearer {auth_token}",
                    "content-type": "application/json",
                },
                json={
                    "messages": messages,
                    "stream": True,
                },
                timeout=None,
            ) as response:
                async for chunk in response.aiter_bytes():
                    yield chunk

    return StreamingResponse(response_generator(), media_type="application/json")


# --- Endpoint FastAPI ---
@app.get("/", response_model=dict)
async def root() -> dict:
    """
    Point d'entrée principal de l'API.
    Retourne un message de bienvenue.
    """
    return {"message": "Bienvenue dans l'API Copilot LLM!"}


@app.post("/")
async def query_copilot(request: Request) -> StreamingResponse:
    """
    Endpoint pour interagir avec l'API Copilot en mode streaming.
    """
    return await handle_copilot_query(request)
