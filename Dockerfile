# Règle appliquée: Modularisation - Multi-stage build pour optimiser l'image
FROM python:3.12-slim as builder

# Installation des dépendances systèmes nécessaires
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Installation de Poetry
RUN pip install poetry==1.7.1

# Configuration de Poetry
RUN poetry config virtualenvs.create false

# Copie des fichiers de dépendances
WORKDIR /app
COPY pyproject.toml poetry.lock ./

# Installation des dépendances
RUN poetry install --no-dev --no-interaction

# Stage final
FROM python:3.12-slim

# Installation des dépendances minimales
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copie de l'environnement virtuel depuis le builder
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

WORKDIR /app

# Copie du code source
COPY app app/
COPY scripts scripts/

# Variables d'environnement
ENV PYTHONPATH=/app
ENV PORT=8000
ENV HOST=0.0.0.0

# Exposition du port
EXPOSE 8000

# Script de démarrage
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]