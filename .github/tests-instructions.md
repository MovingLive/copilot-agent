
# Bonnes Pratiques pour les Tests Unitaires et d'Intégration avec Pytest

Ce guide résume les bonnes pratiques pour écrire des tests unitaires et d'intégration efficaces en Python en utilisant Pytest. Ces recommandations sont adaptées pour maximiser l'efficacité et la lisibilité des tests.

Exécuter les tests unitaires en utilisant l'outil de test intégré de VSCode.

---

## 1. **Structure des Tests**

- **Nommer les tests de manière explicite** :
  - Exemple : `test_user_creation_returns_201`.
- **Organiser les fichiers** :
  - Placez vos tests dans un dossier `tests` à la racine du projet.
  - Utilisez des fichiers comme `test_models.py`, `test_routers.py`.

---

## 2. **Utilisation des Fixtures**

- **Créer des Fixtures Réutilisables** :
  - Placez les fixtures communes dans un fichier `conftest.py` pour un accès global.
- **Définir le Scope des Fixtures** :
  - `scope="function"` : Réinitialiser entre chaque test.
  - `scope="session"` : Partager entre plusieurs tests pour éviter de répéter des opérations coûteuses (ex. création de tables).

### Exemple :
```python
@pytest.fixture(scope="function")
def session(engine):
    connection = engine.connect()
    transaction = connection.begin()
    session = Session(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()
```

---

## 3. **Tests Paramétrés**

- **Utilisez `@pytest.mark.parametrize` pour tester plusieurs cas avec la même fonction.**
- Exemple :
```python
@pytest.mark.parametrize(
    "input_data,expected_result",
    [
        ([1, 2, 3], 6),
        ([10, 20], 30),
        ([], 0),
    ],
)
def test_sum(input_data, expected_result):
    assert sum(input_data) == expected_result
```

---

## 4. **Isolation des Tests**

- **Réinitialiser les Données de Test** :
  - Utilisez des transactions pour isoler les tests sans recréer la base :
    ```python
    @pytest.fixture(scope="function")
    def clean_database(session):
        yield
        session.rollback()
    ```
- **Mocker les dépendances externes** :
  - Utilisez `unittest.mock` pour remplacer les appels réseau, fichiers ou services tiers.

---

## 5. **Tests d'Intégration**

- **Utilisez une base de données en mémoire pour les tests rapides** :
  - Exemple : `sqlite:///:memory:`.
- **Surchargez les dépendances** :
  - Exemple pour FastAPI :
    ```python
    @pytest.fixture(scope="function")
    def client(session):
        def override_get_session():
            return session

        app.dependency_overrides[get_session] = override_get_session
        client = TestClient(app)
        yield client
        app.dependency_overrides.clear()
    ```

---

## 6. **Organisation des Données**

- **Préparer des Données Réutilisables** :
  - Centralisez la logique dans des fixtures pour éviter la duplication :
    ```python
    @pytest.fixture
    def sample_user_data():
        return {"name": "John Doe", "email": "john@example.com"}
    ```

---

## 7. **Bonnes Pratiques Générales**

- **Tester un Comportement à la Fois** :
  - Gardez les tests simples et spécifiques.
- **Éviter les Effets de Bord** :
  - Les tests doivent être indépendants les uns des autres.
- **Inclure des Assertions Lisibles** :
  - Exemple : `assert "success" in response.json()`.
- **Couvrir les Cas Limites** :
  - Exemples : données vides, erreurs de validation, permissions insuffisantes.

---

## 8. **Tests Automatisés et CI/CD**

- **Intégrer Pytest dans vos Pipelines CI/CD** :
  - Commande typique : `pytest --maxfail=1 --disable-warnings`.
- **Utiliser des Plugins** :
  - Exemples : `pytest-cov` pour la couverture de code, `pytest-mock` pour les mocks.

---

## 9. **Exemple Complet de Test d'Intégration FastAPI**

Voici un exemple intégrant toutes les bonnes pratiques :
```python
import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.models import User

@pytest.fixture(scope="function")
def client(session):
    def override_get_session():
        return session

    app.dependency_overrides[get_session] = override_get_session
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()

def test_create_user(client):
    response = client.post("/users/", json={"name": "Alice", "email": "alice@example.com"})
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Alice"
    assert data["email"] == "alice@example.com"
```

---

Suivez ces principes pour créer des tests robustes, lisibles et maintenables, adaptés à tous vos besoins en Python avec Pytest.
