# GitHub Copilot Extension

## API FastAPI

### Objectif principal de l'API

API RESTful avec FastAPI pour un assistant GitHub Copilot. Elle récupère des embeddings issus de fichiers Markdown via FAISS et Chroma DB, enrichit les requêtes utilisateurs avec des données contextuelles, et utilise un LLM pour générer des réponses. Un script automatisé met à jour quotidiennement l'index et synchronise les données vers AWS S3. L'API gère automatiquement la traduction entre le français et l'anglais pour assurer une meilleure correspondance entre les questions et la documentation.

### Architecture optimisée du système RAG

L'API REST suit un pipeline RAG (Retrieval-Augmented Generation) hautement optimisé, avec les caractéristiques suivantes :

1. **Segmentation intelligente des documents Markdown** : Analyse de structure avec chevauchement intelligent (50 caractères) et déduplication des segments trop similaires (seuil 0.85).

2. **Détection et traduction automatique** :
   - Détection automatique de la langue des questions entrantes
   - Traduction bidirectionnelle français-anglais avec M2M100 (version optimisée 418M)
   - Cache LRU pour les modèles de traduction et les résultats fréquents
   - Gestion robuste des erreurs pour assurer la continuité du service

3. **Indexation vectorielle performante** :
   - Sélection automatique du type d'index FAISS optimal selon la taille des données (FlatL2, IVF ou HNSW)
   - Paramètres de recherche configurables via FAISS_LANG et autres variables d'environnement
   - Optimisation des requêtes par langue pour une meilleure pertinence

4. **Cache vectoriel multi-niveau** : Mécanisme LRU (Least Recently Used) pour les embeddings et les résultats de recherche, avec TTL (Time-To-Live) de 2 heures.

5. **Prompts structurés pour le LLM** :
   - Extraction et priorisation automatique des faits clés
   - Formatage contexte en sections pertinentes
   - Conservation de la langue originale de l'utilisateur dans les réponses

6. **Récupération de données pertinentes** :
   - Recherche de documents similaires via FAISS avec paramètres optimisés
   - Traduction automatique du contexte si nécessaire
   - Équilibrage entre précision et performance

7. **Génération de réponses enrichies** :
   - Interrogation du LLM de GitHub Copilot avec contexte enrichi
   - Gestion intelligente des langues source et cible
   - Support multilingue transparent pour l'utilisateur

### Toolstack

- **FastAPI**: Framework API RESTful
- **Python**: Langage de programmation principal
- **Poetry**: Gestion des dépendances et packaging
- **pytest**: Tests unitaires et d'intégration
- **Pydantic V2**: Validation des données et sérialisation
- **uvicorn**: Serveur ASGI
- **SentenceTransformers**: Génération d'embeddings vectoriels
- **M2M100**: Traduction multilingue (facebook/m2m100_418M)
- **langdetect**: Détection de langue avec stabilisation
- **Chroma DB**: Base de données vectorielle
- **FAISS**: Base de données vectorielle optimisée
- **Boto3**: Interaction avec AWS S3
- **StreamingResponse**: Réponses en streaming
- **GitHub API**: Authentification et informations utilisateur
- **Copilot LLM API**: Génération de réponses contextuelles
- **Logging**: Journalisation et débogage
- **GitHub Actions**: CI/CD

### Diagrame de séquence sur demarrage du serveur

```mermaid
sequenceDiagram
    participant FastAPI
    participant Lifespan
    participant EmbeddingService
    participant FAISSService
    participant VectorCache
    participant UpdateThread
    participant CORSMiddleware
    participant Routers

    FastAPI->>Lifespan: Démarre le gestionnaire de cycle de vie

    Lifespan->>VectorCache: Initialise le cache vectoriel
    VectorCache->>VectorCache: Établit le cache LRU (2000 entrées, TTL=7200s)

    Lifespan->>EmbeddingService: Initialise le singleton
    EmbeddingService->>EmbeddingService: Charge le modèle SentenceTransformer
    EmbeddingService->>VectorCache: Configure lien vers le cache
    alt Erreur de chargement du modèle
        EmbeddingService-->>Lifespan: Erreur d'initialisation
        Lifespan-->>FastAPI: Propage l'erreur
    end

    Lifespan->>FAISSService: load_index()
    FAISSService->>FAISSService: Charge index.faiss et metadata.json
    FAISSService->>FAISSService: Configure paramètres optimaux (nprobe/efSearch)
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
- **VectorCache**: Service singleton qui gère le cache d'embeddings et de résultats.
- **EmbeddingService**: Service singleton qui charge le modèle SentenceTransformer et utilise le cache.
- **FAISSService**: Charge l'index vectoriel optimisé et les métadonnées associées.
- **UpdateThread**: Thread daemon qui met à jour périodiquement l'index FAISS.
- **CORSMiddleware**: Configure les politiques CORS.
- **Routers**: Monte les routeurs pour les endpoints health et copilot.

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

- Ouvrir le port 8000 dans l'onglet `PORTS` de VSCode
- Rendre l'URL Public
- Copier l'URL publique et la coller dans la GitHub App mode Copilot Extension

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

### Configuration du système

Le système peut être configuré via des variables d'environnement :

- **FAISS_LANG**: Langue principale de la documentation (default: "en")
- **LOG_LEVEL**: Niveau de journalisation (default: "INFO")
- **CORS_ORIGINS**: Origines autorisées pour CORS
- **ENV**: Environnement d'exécution (local/production)

Exemple de configuration dans `.env` :

```bash
FAISS_LANG=en
LOG_LEVEL=INFO
ENV=local
```

### Pipeline de traduction

Le système intègre un pipeline de traduction robuste :

1. **Détection de langue** :
   - Utilisation de langdetect avec seed fixe pour la stabilité
   - Validation supplémentaire avec dictionnaire de mots courants
   - Gestion des cas particuliers et des erreurs

2. **Traduction** :
   - Modèle M2M100 optimisé (418M paramètres)
   - Mise en cache du modèle avec @lru_cache
   - Traitement asynchrone pour les performances

3. **Optimisation** :
   - Cache LRU pour les résultats de traduction
   - Vérification de nécessité de traduction
   - Conservation du texte original en cas d'erreur

4. **Validation** :
   - Tests unitaires complets
   - Gestion des cas limites
   - Monitoring des performances
