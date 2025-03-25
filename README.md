# Copilot Agent avec RAG

Service FastAPI utilisant FAISS et Copilot LLM pour répondre aux questions des développeurs en se basant sur la documentation GitHub.

## 🚀 Fonctionnalités

- RAG (Retrieval Augmented Generation) avec FAISS pour la recherche sémantique
- Intégration avec GitHub Copilot LLM
- Support pour ChromaDB comme alternative à FAISS
- Synchronisation avec AWS S3 pour la persistance
- API RESTful avec FastAPI
- Tests unitaires et d'intégration complets
- CI/CD avec GitHub Actions

## 📋 Prérequis

- Python 3.10+
- Poetry pour la gestion des dépendances
- Git
- Docker et Docker Compose (optionnel)

## 🛠️ Installation

1. Cloner le repository :

```bash
git clone https://github.com/votre_utilisateur/copilot-agent.git
cd copilot-agent
```

2. Installer les dépendances avec Poetry :

```bash
poetry install
```

3. Configurer les variables d'environnement :

```bash
cp .env.example .env
# Éditer .env avec vos configurations
```

## 🚦 Démarrage

### Développement local avec Poetry

```bash
poetry run uvicorn app.main:app --reload
```

### Avec Docker Compose

```bash
docker-compose up --build
```

## 🧪 Tests

Exécuter les tests avec pytest :

```bash
poetry run pytest
```

Vérifier le typage avec mypy :

```bash
poetry run mypy app
```

Linting avec ruff :

```bash
poetry run ruff check app tests
```

## 📚 Documentation API

Une fois le serveur démarré, la documentation OpenAPI est disponible à :

- <http://localhost:8000/docs> (Swagger UI)
- <http://localhost:8000/redoc> (ReDoc)

## 🔄 Mise à jour de l'index

### Index FAISS

Pour mettre à jour l'index FAISS avec de nouveaux documents :

```bash
poetry run python scripts/update_faiss.py
```

### Index ChromaDB

Pour utiliser ChromaDB à la place de FAISS :

```bash
poetry run python scripts/update_chroma.py
```

## 🏗️ Structure du Projet

```
.
├── app/                    # Code source principal
│   ├── api/               # Endpoints FastAPI
│   ├── core/             # Configuration et fonctions core
│   ├── services/         # Services métier
│   └── utils/            # Utilitaires
├── scripts/              # Scripts de maintenance
├── tests/               # Tests
└── output/              # Fichiers générés
```

## 🤝 Contribution

1. Fork le projet
2. Créer une branche (`git checkout -b feature/nouvelle-fonctionnalite`)
3. Commit les changements (`git commit -am 'Ajout de nouvelle fonctionnalité'`)
4. Push la branche (`git push origin feature/nouvelle-fonctionnalite`)
5. Créer une Pull Request

## 📄 Licence

MIT

## 🔧 Configuration

### Variables d'environnement

| Variable | Description | Default |
|----------|-------------|---------|
| ENV | Environnement d'exécution | local |
| REPO_URL | URL du dépôt GitHub | - |
| COPILOT_TOKEN | Token GitHub Copilot | - |
| AWS_REGION | Région AWS | ca-central-1 |
| S3_BUCKET_NAME | Nom du bucket S3 | mon-bucket-faiss |

### Configuration FAISS

L'index FAISS utilise le modèle `all-MiniLM-L6-v2` avec une dimension de 384.

### Monitoring

Un endpoint `/health` est disponible pour le monitoring.
