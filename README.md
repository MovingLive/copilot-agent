# GitHub Copilot Extension

## API FastAPI

### Objectif principal de l'API

Créer une API RESTful pour interagir avec le modèle d'embedding et la base de données vectorielle.

### Contexte de l'API

L'API est construite avec FastAPI. Elle permet de gérer les requêtes HTTP, d'interagir avec la base de données vectorielle pour récupérer les embeddings et de fournir cela en contexte au LLM de GitHub Copilot.

### Architecture de l'API

L’API REST demandée suit un pipeline RAG (Retrieval-Augmented Generation). Lorsqu’elle reçoit une requête JSON contenant une question de l’utilisateur (ainsi qu’un contexte optionnel provenant de GitHub Copilot Chat), elle procède en deux étapes :

1. **Récupération de données pertinentes liées à la question à l’aide d’une base de vecteurs (FAISS) contenant des embeddings.**

2. **Génération de la réponse** en interrogeant le Large Language Model (LLM) de GitHub Copilot en lui fournissant la question enrichie du contexte récupéré. Enfin, la réponse du LLM est renvoyée au client via l’API FastAPI.

### Toolstrack

- **FastAPI**: Utilisé pour créer l'API RESTful.
- **Python**: Langage de programmation utilisé pour le développement de l'API.
- **Poetry**: Outil de gestion des dépendances et de packaging pour Python.
- **pytest**: Utilisé pour les tests unitaires et d'intégration.
- **Pydantic**: Utilisé pour la validation des données et la sérialisation.
- **uvicorn**: Serveur ASGI pour exécuter l'application FastAPI.
- **SentenceTransformers**: Utilisé pour générer des embeddings vectoriels à partir de texte.
- **Chroma DB**: Base de données vectorielle pour le stockage et la recherche d'embeddings.
- **FAISS**: Base de données vectorielle pour le stockage et la recherche d'embeddings.
- **Boto3**: Utilisé pour interagir avec AWS S3, bien que cela ne soit pas directement visible dans cet endpoint, mais dans les fonctions auxiliaires.
- **StreamingResponse**: Utilisé pour envoyer des réponses de streaming à l'utilisateur.
- **GitHub API**: Utilisée pour authentifier l'utilisateur via le token GitHub (x-github-token) et récupérer des informations sur l'utilisateur.
- **Copilot LLM API**: API appelée pour générer une réponse basée sur la question et le contexte fourni.
- **Logging**: Utilisé pour enregistrer des informations sur le traitement des requêtes et le débogage.
- **GitHub Actions**: Utilisé pour l'intégration continue et le déploiement continu (CI/CD).

### Diagrame de séquence sur demarrage du serveur

```mermaid
sequenceDiagram
    participant FastAPI
    participant Lifespan
    participant EmbeddingService
    participant FAISSService
    participant UpdateThread
    participant CORSMiddleware
    participant Routers

    FastAPI->>Lifespan: Démarre le gestionnaire de cycle de vie

    Lifespan->>EmbeddingService: Initialise le singleton
    EmbeddingService->>EmbeddingService: Charge le modèle SentenceTransformer
    alt Erreur de chargement du modèle
        EmbeddingService-->>Lifespan: Erreur d'initialisation
        Lifespan-->>FastAPI: Propage l'erreur
    end

    Lifespan->>FAISSService: load_index()
    FAISSService->>FAISSService: Charge index.faiss et metadata.json
    alt Erreur de chargement FAISS
        FAISSService-->>Lifespan: Erreur d'initialisation
        Lifespan-->>FastAPI: Propage l'erreur
    end

    Lifespan->>UpdateThread: Démarre thread de mise à jour
    UpdateThread->>FAISSService: update_periodically() (async)

    FastAPI->>CORSMiddleware: Configure CORS
    FastAPI->>Routers: Monte les routeurs (health, copilot)

    FastAPI-->>Client: Serveur prêt
```

Explications :

- **FastAPI**: Démarre l'application et gère le cycle de vie.
- **Lifespan**: Gestionnaire de cycle de vie qui initialise les services essentiels.
- **EmbeddingService**: Service singleton qui charge le modèle SentenceTransformer.
- **FAISSService**: Charge l'index vectoriel et les métadonnées associées.
- **UpdateThread**: Thread daemon qui met à jour périodiquement l'index FAISS.
- **CORSMiddleware**: Configure les politiques CORS.
- **Routers**: Monte les routeurs pour les endpoints health et copilot.

Les erreurs critiques lors de l'initialisation empêchent le démarrage du serveur.

### Diagrame de séquence sur appel de l'API

```mermaid
sequenceDiagram
    participant Client
    participant FastAPI
    participant CopilotRouter
    participant FAISSService
    participant EmbeddingService
    participant CopilotService
    participant GitHubAPI

    Client->>FastAPI: POST /copilot (x-github-token + messages)
    FastAPI->>CopilotRouter: handle_copilot_query(request)

    alt Token manquant
        CopilotRouter-->>Client: 401 Missing authentication token
    end

    CopilotRouter->>CopilotRouter: Extraire query des messages
    alt Messages vides
        CopilotRouter-->>Client: 400 Messages manquants
    end

    CopilotRouter->>FAISSService: retrieve_similar_documents(query, k=5)
    FAISSService->>EmbeddingService: generate_query_vector(query)
    EmbeddingService-->>FAISSService: query_vector
    FAISSService->>FAISSService: Recherche dans l'index
    FAISSService-->>CopilotRouter: documents similaires

    CopilotRouter->>CopilotRouter: Formater le contexte
    CopilotRouter->>GitHubAPI: GET /user (auth_token)
    alt Authentification réussie
        GitHubAPI-->>CopilotRouter: user_login
    else Authentification échouée
        CopilotRouter-->>Client: 401 Invalid GitHub token
    end

    CopilotRouter->>CopilotService: format_copilot_messages(query, context, user)
    CopilotService-->>CopilotRouter: messages formatés
    CopilotRouter->>CopilotService: generate_streaming_response(messages, token)
    CopilotRouter->>FastAPI: StreamingResponse
    FastAPI-->>Client: Réponse en streaming
```

Explications :

- **Client**: Envoie une requête POST avec un token GitHub et des messages.
- **FastAPI**: Délègue la requête au router Copilot.
- **CopilotRouter**:
  - Vérifie la présence du token et des messages.
  - Extrait la dernière question des messages.
  - Gère la recherche de documents similaires via FAISS.
  - Formate le contexte avec les documents trouvés.
  - Obtient les informations utilisateur via GitHub.
  - Formate les messages pour Copilot.
  - Configure la réponse en streaming.
- **FAISSService**:
  - Utilise EmbeddingService pour vectoriser la requête.
  - Recherche les documents similaires dans l'index.
- **EmbeddingService**: Génère les vecteurs d'embedding avec SentenceTransformer.
- **CopilotService**: Formate les messages et gère l'interaction avec l'API Copilot.
- **GitHubAPI**: Authentifie l'utilisateur et fournit ses informations.
- **FastAPI**: Retourne la réponse en streaming au client.

### Installation locale

- Installer les dépendances du projet avec Poetry.

```bash
pip install poetry
poetry install
poetry env activate
```

- Créer un fichier `.env` à la racine du projet avec les variables d'environnement nécessaires.

> Prendre exempl de `.env.example` et le renommer en `.env`.

- Lancer l'application FastAPI.

Utiliser le launcher de VSCode pour démarrer le serveur.
Puis accéder à l'API via `http://localhost:8000/` pour confirmer que tout va bien.

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
