import asyncio
import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import boto3
import faiss
import httpx
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict

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
    load_faiss_index()
    # Lancer la mise à jour périodique de l'index dans un thread de fond
    threading.Thread(target=update_faiss_index_periodically, daemon=True).start()
    logger.info("Service de mise à jour de l'index lancé en arrière-plan.")

    yield
    logger.info("Arrêt de l'application. Nettoyage des ressources...")


# --- Instance FastAPI ---
app = FastAPI(lifespan=lifespan)


# --- Modèles Pydantic ---
class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class QueryResponse(BaseModel):
    answer: str

    model_config = ConfigDict(from_attributes=True)


# --- Fonctions utilitaires ---
def load_faiss_index() -> None:
    """
    Télécharge et charge l'index FAISS depuis AWS S3 ou le répertoire local.
    Optionnellement, charge aussi un mapping vers les documents.
    """
    global FAISS_INDEX, document_store
    try:
        local_faiss_path = None
        if is_local_environment():
            # Charger depuis le répertoire local
            local_faiss_path = os.path.join(LOCAL_OUTPUT_DIR, FAISS_INDEX_FILE)
            if not os.path.exists(local_faiss_path):
                logger.warning(
                    "Index FAISS introuvable dans le répertoire local, création d'un index vide."
                )
                local_faiss_path = f"/tmp/{FAISS_INDEX_FILE}"
                # Créer un index vide si aucun n'existe
                empty_index = faiss.IndexFlatL2(128)  # Dimension 128 par défaut
                faiss.write_index(empty_index, local_faiss_path)
        else:
            # Télécharger l'index depuis S3 vers un chemin temporaire
            local_faiss_path = f"/tmp/{FAISS_INDEX_FILE}"
            s3_client = boto3.client("s3", region_name=AWS_REGION)
            s3_client.download_file(S3_BUCKET_NAME, FAISS_INDEX_FILE, local_faiss_path)
            logger.info("Index FAISS téléchargé depuis S3.")

        # Charger l'index FAISS
        FAISS_INDEX = faiss.read_index(local_faiss_path)
        logger.info("Index FAISS chargé en mémoire.")

        # Optionnel : charger un mapping document associé à l'index.
        try:
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

            with open(local_metadata_path, "r", encoding="utf-8") as f:
                document_store = json.load(f)
            logger.info("Mapping des documents chargé.")
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
    Fonction fictive pour générer un embedding à partir d'un texte.
    Remplace-la par ton modèle d'embedding (par exemple SentenceTransformers).
    Ici, on retourne un vecteur aléatoire correspondant à la dimension de l'index.
    """
    # Dimension fixe de 384 pour garantir la cohérence
    dim = 384

    # Utiliser le même seed pour le même texte pour garantir la cohérence
    rng = np.random.RandomState(abs(hash(text)) % (2**32))
    vector = rng.rand(dim).astype("float32")
    return vector.tolist()


def retrieve_similar_documents(query: str, k: int = 3) -> List[dict]:
    """
    Recherche dans l'index FAISS les k documents les plus proches de la requête.
    Retourne une liste de dictionnaires représentant les documents.
    """
    try:
        if not query:
            raise ValueError("Query cannot be empty")

        if FAISS_INDEX is None:
            logger.error("Index FAISS non chargé.")
            return []

        # Générer l'embedding de la requête
        query_vector = np.array([embed_text(query)]).astype("float32")

        if query_vector.size == 0:
            raise ValueError("Failed to generate embedding for query")

        # Logs pour diagnostiquer l'AssertionError
        logger.info("Query vector shape: %s", query_vector.shape)
        logger.info("Query vector type: %s", query_vector.dtype)
        logger.info("FAISS index size: %d", FAISS_INDEX.ntotal)
        logger.info("FAISS index dimension: %d", FAISS_INDEX.d)

        # Vérification de compatibilité des dimensions
        if query_vector.shape[1] != FAISS_INDEX.d:
            logger.error(
                "Incompatibilité de dimensions: vecteur de requête (%d) != index FAISS (%d)",
                query_vector.shape[1],
                FAISS_INDEX.d,
            )
            # Ajuster dynamiquement la dimension du vecteur de requête
            new_query_vector = np.zeros((1, FAISS_INDEX.d), dtype="float32")
            min_dim = min(query_vector.shape[1], FAISS_INDEX.d)
            new_query_vector[0, :min_dim] = query_vector[0, :min_dim]
            query_vector = new_query_vector
            logger.info(
                "Vecteur de requête redimensionné à la dimension %d", FAISS_INDEX.d
            )

        try:
            logger.info("Exécution de FAISS_INDEX.search avec k=%d", k)
            distances, indices = FAISS_INDEX.search(query_vector, k)
            logger.info(
                "Recherche FAISS réussie. Distances shape: %s, Indices shape: %s",
                distances.shape,
                indices.shape,
            )
        except AssertionError as ae:
            logger.error(
                "AssertionError dans FAISS_INDEX.search: %s",
                str(ae) if str(ae) else "Assertion vide",
            )
            logger.error("Détails de l'index FAISS: %s", str(FAISS_INDEX))
            logger.error("Est-ce que l'index est vide? %s", FAISS_INDEX.ntotal == 0)
            logger.error(
                "Vecteur de requête (premiers éléments): %s", query_vector.flatten()[:5]
            )
            raise HTTPException(
                status_code=500, detail="Erreur d'assertion lors de la recherche FAISS"
            ) from ae

        # Logs pour déboguer document_store
        logger.debug("Structure de document_store: %s", type(document_store))
        logger.debug("Taille de document_store: %d", len(document_store))
        logger.debug(
            "Clés disponibles: %s",
            list(document_store.keys())[:5]
            if isinstance(document_store, dict)
            else "Non dict",
        )

        results = []
        for idx in indices[0]:
            # Accéder au document en utilisant l'index comme clé de chaîne
            if idx >= 0:
                doc_key = str(idx)
                if doc_key in document_store:
                    results.append(document_store[doc_key])
                else:
                    logger.debug("Index non trouvé dans document_store: %s", idx)
            else:
                logger.debug("Index invalide ignoré: %s", idx)

        if not results:
            logger.warning("No matching documents found for query: %s", query)

        return results

    except ValueError as ve:
        logger.error("Validation error in retrieve_similar_documents: %s", ve)
        raise HTTPException(status_code=400, detail=str(ve)) from ve
    except Exception as e:
        logger.error("Error in retrieve_similar_documents: %s", e)
        logger.error("Type d'exception: %s", type(e).__name__)
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
    docs = retrieve_similar_documents(message + " " + additional_context, k=3)
    context_text = "\n".join([doc.get("content", "") for doc in docs])
    logger.info("Contexte récupéré via FAISS.")

    # Appeler l'API Copilot LLM
    answer = call_copilot_llm(message, context_text, auth_token)

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
            "content": "Tu es un assistant qui fournit une assistance technique et tres pédagogue.",
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