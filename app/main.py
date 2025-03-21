import json
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import boto3
import faiss
import numpy as np
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# --- Chargement des variables d'environnement ---
load_dotenv()

# --- Configuration ---
S3_BUCKET = os.getenv("S3_BUCKET", "mon-bucket-faiss")
FAISS_KEY = os.getenv("FAISS_KEY", "index.faiss")
AWS_REGION = os.getenv("AWS_REGION", "canada-central-1")
COPILOT_API_URL = os.getenv(
    "COPILOT_API_URL", "https://api.githubcopilot.com/chat/completions"
)
COPILOT_TOKEN = os.getenv("COPILOT_TOKEN", "")

# --- Variables globales ---
FAISS_INDEX = None
# Optionnel : stocke une liste de documents associés aux vecteurs de l'index.
# Chaque document peut être un dictionnaire, par exemple : {"id": 0, "content": "texte ..."}
document_store = []

# --- Logger ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("copilot_api")


# --- Événement de démarrage ---
@asynccontextmanager
async def lifespan(_: FastAPI):
    logger.info("Démarrage de l'application et chargement de l'index FAISS...")
    load_faiss_index()
    # Lancer la mise à jour périodique de l'index dans un thread de fond
    threading.Thread(target=update_faiss_index_periodically, daemon=True).start()
    logger.info("Service de mise à jour de l'index lancé en arrière-plan.")

    yield
    logger.info("Arrêt de l'application. Nettoyage des ressources...")


# --- Instance FastAPI ---
# app = FastAPI(lifespan=lifespan)
app = FastAPI()


# --- Modèles Pydantic ---
class QueryRequest(BaseModel):
    question: str
    context: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str


# --- Fonctions utilitaires ---


def load_faiss_index() -> None:
    """
    Télécharge et charge l'index FAISS depuis AWS S3.
    Optionnellement, charge aussi un mapping vers les documents.
    """
    global FAISS_INDEX, document_store
    try:
        s3_client = boto3.client("s3", region_name=AWS_REGION)
        # Télécharger l'index depuis S3 vers un chemin temporaire
        local_faiss_path = f"/tmp/{FAISS_KEY}"
        s3_client.download_file(S3_BUCKET, FAISS_KEY, local_faiss_path)
        logger.info("Index FAISS téléchargé depuis S3.")

        # Charger l'index FAISS
        FAISS_INDEX = faiss.read_index(local_faiss_path)
        logger.info("Index FAISS chargé en mémoire.")

        # Optionnel : charger un mapping document associé à l'index.
        # On suppose qu'un fichier JSON (ex: index.faiss.json) existe dans le bucket.
        local_mapping_path = local_faiss_path + ".json"
        try:
            s3_client.download_file(S3_BUCKET, FAISS_KEY + ".json", local_mapping_path)
            with open(local_mapping_path, "r", encoding="utf-8") as f:
                document_store = json.load(f)
            logger.info("Mapping des documents chargé.")
        except Exception:
            logger.warning(
                "Mapping des documents introuvable, utilisation d'une liste vide."
            )
            document_store = []
    except Exception as e:
        logger.error("Erreur lors du chargement de l'index FAISS : %s", e)


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
    Ici, on retourne un vecteur aléatoire de dimension 128.
    """
    np.random.seed(abs(hash(text)) % (2**32))
    return np.random.rand(128).tolist()


def retrieve_similar_documents(query: str, k: int = 3) -> List[dict]:
    """
    Recherche dans l'index FAISS les k documents les plus proches de la requête.
    Retourne une liste de dictionnaires représentant les documents.
    """
    if FAISS_INDEX is None:
        logger.error("Index FAISS non chargé.")
        return []
    # Générer l'embedding de la requête
    query_vector = np.array([embed_text(query)]).astype("float32")
    distances, indices = FAISS_INDEX.search(query_vector, k)
    results = []
    for idx in indices[0]:
        if idx < len(document_store):
            results.append(document_store[idx])
    return results


def call_copilot_llm(question: str, context_text: str) -> str:
    """
    Appelle l'API GitHub Copilot LLM en fournissant une question et le contexte récupéré.
    Retourne la réponse générée.
    """
    headers = {
        "Authorization": f"Bearer {COPILOT_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "messages": [
            {"role": "system", "content": f"Contexte:\n{context_text}"},
            {"role": "user", "content": question},
        ]
    }
    try:
        response = requests.post(
            COPILOT_API_URL, json=payload, headers=headers, timeout=30
        )
        response.raise_for_status()
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        return answer
    except Exception as e:
        logger.error("Erreur lors de l'appel à l'API Copilot LLM : %s", e)
        raise HTTPException(
            status_code=500, detail="Erreur lors de l'appel au service Copilot LLM."
        ) from e


# --- Endpoint FastAPI ---

@app.get("/", response_model=dict)
async def root():
    """
    Point d'entrée principal de l'API.
    Retourne un message de bienvenue.
    """
    return {"message": "Bienvenue dans l'API Copilot LLM!"}

@app.post("/query", response_model=QueryResponse)
async def query_copilot(request: QueryRequest) -> QueryResponse:
    """
    Gère la requête de l'utilisateur et retourne la réponse générée par l'API Copilot LLM.
    """
    question = request.question
    additional_context = request.context or ""
    logger.info("Requête reçue : %s", question)
    # Récupérer des documents similaires à la question (en combinant la question et le contexte additionnel)
    docs = retrieve_similar_documents(question + " " + additional_context, k=3)
    # Concaténer les contenus des documents récupérés
    context_text = "\n".join([doc.get("content", "") for doc in docs])
    logger.info("Contexte récupéré via FAISS.")
    # Appeler l'API Copilot LLM avec la question et le contexte
    answer = call_copilot_llm(question, context_text)
    logger.info("Réponse générée par l'API Copilot LLM.")
    return QueryResponse(answer=answer)
