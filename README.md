# GitHub Copilot Extension

## API FastAPI

### Objectif principal de l'API

Créer une API RESTful pour interagir avec le modèle d'embedding et la base de données vectorielle.

### Contexte de l'API

L'API est construite avec FastAPI, un framework moderne et rapide pour la création d'APIs en Python. Elle permet de gérer les requêtes HTTP, d'interagir avec la base de données et de servir les embeddings.

### Architecture de l'API

L’API REST demandée suit un pipeline RAG (Retrieval-Augmented Generation). Lorsqu’elle reçoit une requête JSON contenant une question de l’utilisateur (ainsi qu’un contexte optionnel provenant de GitHub Copilot Chat), elle procède en deux étapes :

1. **Récupération de données pertinentes liées à la question à l’aide d’une base de vecteurs (FAISS) contenant des embeddings.**
2. **Génération de la réponse** en interrogeant le Large Language Model (LLM) de GitHub Copilot en lui fournissant la question enrichie du contexte récupéré. Enfin, la réponse du LLM est renvoyée au client via l’API FastAPI.

### Toolstrack

- **Python**: Langage de programmation utilisé pour le développement de l'API.
- **Poetry**: Outil de gestion des dépendances et de packaging pour Python.
- **GitHub Actions**: Utilisé pour l'intégration continue et le déploiement continu (CI/CD).
- **pytest**: Utilisé pour les tests unitaires et d'intégration.
- **FastAPI**: Utilisé pour créer l'API RESTful.
- **Pydantic**: Utilisé pour la validation des données et la sérialisation.
- **uvicorn**: Serveur ASGI pour exécuter l'application FastAPI.
- **Chroma DB**: Base de données vectorielle pour le stockage et la recherche d'embeddings.
- **FAISS**: Librairie de Facebook pour la recherche efficace d'embeddings.
- **Docker**: Utilisé pour containeriser l'application et faciliter le déploiement.

### Installation locale

- Installer les dépendances du projet avec Poetry.

```bash
pip install poetry
poetry install
poetry env activate
```

- Créer un fichier `.env` à la racine du projet avec les variables d'environnement nécessaires.

```bash
AWS_REGION=<your_aws_region>
COPILOT_API_URL=https://api.githubcopilot.com/chat/completions
COPILOT_TOKEN=<your_github_token>
ENV=local
REPO_DIR=<your_local_repo_dir>
REPO_URL=<your_repo_url>
S3_BUCKET_NAME=<your_s3_bucket>
FAISS_METADATA_FILE=id_mapping.json
FAISS_INDEX_FILE=index.faiss
```

- Lancer l'application FastAPI.

Utiliser le launcher de VSCode pour démarrer le serveur.
Puis accéder à l'API via `http://localhost:8000/docs` pour voir la documentation interactive de l'API.

### Vérification de l'état de l'API

Vous pouvez vérifier si l'API fonctionne correctement en envoyant une requête de test à l'un des points de terminaison définis.

`http://localhost:8000/`

## Script Update chroma

### Objectif principal du script

Mettre à jour quotidiennement (toutes les 24 heures) une base de données vectorielle Chroma DB indexant la documentation (environ 150 Mo de fichiers Markdown) pour servir un agent GitHub Copilot destiné à des centaines de développeurs.

### Contexte du script

Pour des fins de tests, 2 technologies utilisées:

- **Chroma DB**: Base de données vectorielle pour le stockage et la recherche d'embeddings. `/scripts/update_chroma.py`
- **FAISS**: Librairie de Facebook pour la recherche efficace d'embeddings. `/scripts/update_faiss.py`

### Architecture du script

- **Extraction & Pré-traitement:** Un script Python qui clônera le dépôt GitHub, lira et traitera les 100 fichiers Markdown pour les diviser en segments pertinents.
- **Génération des embeddings:** Pour chaque segment, on génère un vecteur d’embedding. Même si votre agent utilisera le LLM de Copilot pour répondre aux requêtes, il faut disposer d’un index efficace ; un modèle d’embedding léger et performant (comme « all-MiniLM-L6-v2 » par exemple) peut être utilisé pour obtenir une bonne qualité de recherche.
- **Indexation dans Chroma DB:** Les embeddings (avec leurs métadonnées, par exemple le nom du fichier, le contexte, etc.) sont stockés dans une instance de Chroma DB.
- **Persistance sur AWS S3:** La base vectorielle est persistée dans un dossier local, puis synchronisée vers votre bucket S3 (déjà protégé) pour un stockage centralisé et accessible depuis votre instance AWS.
- **Automatisation via GitHub Actions:** Une action planifiée (cron) déclenche l’exécution quotidienne du script pour mettre à jour la base vectorielle.

### Outputs

#### Le fichier .faiss

Ce fichier contient l'index vectoriel lui-même. C’est un binaire optimisé pour :

- Stocker les vecteurs (embeddings)
- Permettre une recherche rapide de similarité (approximate nearest neighbor search)
- Être chargé rapidement en mémoire pour faire des requêtes

Ce fichier est utilisé directement par FAISS au moment où tu fais une recherche. Il est donc crucial pour l’inférence.

### Le fichier .json

Ce fichier contient les métadonnées associées aux vecteurs. Typiquement :

- L’ID ou la clé de chaque vecteur
- Le contenu original (texte, titre, URL, etc.)
- Toute autre information utile à restituer quand tu fais une recherche

Quand tu fais une requête dans ton moteur de recherche vectoriel, tu obtiens un ou plusieurs vecteurs similaires depuis FAISS (grâce au .faiss), mais pour afficher les résultats à l’utilisateur, tu as besoin de retrouver ce que représentait chaque vecteur → c’est là que le .json est utile.
