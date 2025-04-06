"""Tests pour le module github_discussions_utils."""

import json
import os
from unittest.mock import patch, MagicMock

import pytest
import httpx

from app.core.config import settings
from app.utils.github_discussions_utils import (
    execute_graphql_query,
    extract_repo_info,
    fetch_validated_discussions,
    format_discussions_for_faiss,
    get_validated_discussions_from_repos,
)


class TestGraphQLQuery:
    """Tests pour l'exécution des requêtes GraphQL."""

    @patch("app.core.config.settings.GITHUB_TOKEN", "fake-token")
    @patch("httpx.post")
    def test_execute_graphql_query_success(self, mock_post):
        """Vérifier que la requête GraphQL s'exécute correctement."""
        # Configurer le mock
        mock_response = MagicMock()
        mock_response.json.return_value = {"data": {"test": "success"}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        # Exécuter la fonction
        query = "query { test }"
        variables = {"var": "value"}
        result = execute_graphql_query(query, variables)

        # Vérifier le résultat
        assert result == {"data": {"test": "success"}}
        mock_post.assert_called_once()
        # Vérifier que les arguments de la requête sont corrects
        call_args = mock_post.call_args[1]
        assert call_args["json"]["query"] == query
        assert call_args["json"]["variables"] == variables
        assert call_args["headers"]["Authorization"] == "Bearer fake-token"

    @patch("app.core.config.settings.GITHUB_TOKEN", "")
    def test_execute_graphql_query_no_token(self):
        """Vérifier que l'absence de token génère une erreur."""
        with pytest.raises(ValueError, match="Token GitHub non configuré"):
            execute_graphql_query("query", {})

    @patch("app.core.config.settings.GITHUB_TOKEN", "fake-token")
    @patch("httpx.post")
    def test_execute_graphql_query_http_error(self, mock_post):
        """Vérifier la gestion des erreurs HTTP."""
        # Configurer le mock pour lever une erreur
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "HTTP Error", request=MagicMock(), response=MagicMock()
        )
        mock_response.text = "Error details"
        mock_post.return_value = mock_response

        # Vérifier que l'erreur est bien propagée
        with pytest.raises(httpx.HTTPStatusError):
            execute_graphql_query("query", {})


class TestRepoInfoExtraction:
    """Tests pour l'extraction des informations du dépôt."""

    @pytest.mark.parametrize(
        "repo_url,expected",
        [
            ("https://github.com/owner/repo", ("owner", "repo")),
            ("https://github.com/owner/repo.git", ("owner", "repo")),
            ("http://github.com/owner/repo", ("owner", "repo")),
            ("github.com/owner/repo", ("owner", "repo")),
            ("https://github.com/owner/repo/", ("owner", "repo")),
            ("invalid-url", None),
            ("https://gitlab.com/owner/repo", None),
        ],
    )
    def test_extract_repo_info(self, repo_url, expected):
        """Vérifier l'extraction des informations du dépôt depuis différentes URL."""
        assert extract_repo_info(repo_url) == expected


class TestFetchDiscussions:
    """Tests pour la récupération des discussions GitHub."""

    @patch("app.utils.github_discussions_utils.execute_graphql_query")
    def test_fetch_validated_discussions_success(self, mock_execute_query):
        """Vérifier la récupération des discussions validées."""
        # Définir les données de retour simulées
        mock_discussions_data = {
            "data": {
                "repository": {
                    "discussions": {
                        "nodes": [
                            {
                                "id": "D_1",
                                "title": "Discussion 1",
                                "body": "Description 1",
                                "url": "https://github.com/owner/repo/discussions/1",
                                "answer": {
                                    "id": "A_1",
                                    "body": "Réponse 1",
                                    "author": {"login": "user1"},
                                },
                                "category": {"name": "Q&A"},
                                "labels": {"nodes": [{"name": "bug"}, {"name": "help"}]},
                            },
                            {
                                "id": "D_2",
                                "title": "Discussion 2",
                                "body": "Description 2",
                                "url": "https://github.com/owner/repo/discussions/2",
                                "answer": None,  # Pas de réponse validée
                                "category": {"name": "General"},
                                "labels": {"nodes": []},
                            },
                        ]
                    }
                }
            }
        }

        # Configurer le mock
        mock_execute_query.return_value = mock_discussions_data

        # Exécuter la fonction
        results = fetch_validated_discussions("owner", "repo", 10)

        # Vérifier les résultats
        assert len(results) == 1  # Seule la discussion avec réponse validée doit être retournée
        assert results[0]["id"] == "D_1"
        assert results[0]["title"] == "Discussion 1"
        assert results[0]["answer"] == "Réponse 1"
        assert results[0]["labels"] == ["bug", "help"]

    @patch("app.utils.github_discussions_utils.execute_graphql_query")
    def test_fetch_validated_discussions_empty(self, mock_execute_query):
        """Vérifier le comportement avec des données vides."""
        # Simuler un retour sans discussions
        mock_execute_query.return_value = {"data": {"repository": {"discussions": {"nodes": []}}}}

        # Exécuter la fonction
        results = fetch_validated_discussions("owner", "repo")

        # Vérifier que le résultat est une liste vide
        assert results == []

    @patch("app.utils.github_discussions_utils.execute_graphql_query")
    def test_fetch_validated_discussions_error(self, mock_execute_query):
        """Vérifier la gestion des erreurs."""
        # Simuler une erreur lors de l'exécution de la requête
        mock_execute_query.side_effect = Exception("Test error")

        # Vérifier que la fonction gère l'erreur et retourne une liste vide
        results = fetch_validated_discussions("owner", "repo")
        assert results == []


class TestFormatDiscussions:
    """Tests pour le formatage des discussions pour FAISS."""

    def test_format_discussions_for_faiss(self):
        """Vérifier le formatage des discussions pour l'indexation FAISS."""
        # Données de test
        discussions = [
            {
                "id": "D_1",
                "title": "Test Discussion",
                "body": "Description du test",
                "url": "https://github.com/owner/repo/discussions/1",
                "answer": "Voici la réponse",
                "answer_author": "user1",
                "category": "Q&A",
                "labels": ["bug", "help"],
            }
        ]

        # Formater les discussions
        formatted = format_discussions_for_faiss(discussions)

        # Vérifications
        assert len(formatted) == 1
        file_path, content = formatted[0]
        
        # Vérifier le chemin de fichier
        assert file_path.startswith("github_discussions/Q&A/")
        assert "Test_Discussion" in file_path
        assert file_path.endswith(".md")
        
        # Vérifier le contenu
        assert "# Test Discussion" in content
        assert "Description du test" in content
        assert "Voici la réponse" in content
        assert "user1" in content
        assert "bug, help" in content

    def test_format_discussions_for_faiss_empty(self):
        """Vérifier le comportement avec une liste vide de discussions."""
        formatted = format_discussions_for_faiss([])
        assert formatted == []


class TestGetValidatedDiscussions:
    """Tests pour la récupération des discussions validées à partir de multiples dépôts."""

    @patch("app.utils.github_discussions_utils.extract_repo_info")
    @patch("app.utils.github_discussions_utils.fetch_validated_discussions")
    @patch("app.utils.github_discussions_utils.format_discussions_for_faiss")
    def test_get_validated_discussions_from_repos(
        self, mock_format, mock_fetch, mock_extract
    ):
        """Vérifier la récupération et le formatage des discussions de plusieurs dépôts."""
        # Configuration des mocks
        mock_extract.side_effect = [
            ("owner1", "repo1"),  # Premier dépôt valide
            None,                 # Deuxième dépôt invalide
            ("owner2", "repo2"),  # Troisième dépôt valide
        ]
        
        discussions1 = [{"id": "D1", "title": "Discussion 1"}]
        discussions2 = [{"id": "D2", "title": "Discussion 2"}]
        mock_fetch.side_effect = [discussions1, discussions2]
        
        formatted1 = [("path1", "content1")]
        formatted2 = [("path2", "content2")]
        mock_format.side_effect = [formatted1, formatted2]
        
        # Exécuter la fonction
        results = get_validated_discussions_from_repos([
            "https://github.com/owner1/repo1",
            "invalid-url",
            "https://github.com/owner2/repo2",
        ])
        
        # Vérifications
        assert len(results) == 2
        assert results[0] == ("path1", "content1")
        assert results[1] == ("path2", "content2")
        
        # Vérifier que les fonctions ont été appelées correctement
        assert mock_extract.call_count == 3
        assert mock_fetch.call_count == 2
        mock_fetch.assert_any_call("owner1", "repo1")
        mock_fetch.assert_any_call("owner2", "repo2")
        mock_format.assert_any_call(discussions1)
        mock_format.assert_any_call(discussions2)

    @patch("app.utils.github_discussions_utils.extract_repo_info")
    def test_get_validated_discussions_from_repos_all_invalid(self, mock_extract):
        """Vérifier le comportement quand toutes les URL sont invalides."""
        # Simuler des URL invalides
        mock_extract.return_value = None
        
        # Exécuter la fonction
        results = get_validated_discussions_from_repos(["invalid-url1", "invalid-url2"])
        
        # Vérifier que le résultat est une liste vide
        assert results == []
        assert mock_extract.call_count == 2