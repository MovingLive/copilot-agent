"""Tests unitaires pour le module git_utils."""

import os
import tempfile
from unittest.mock import patch, call, Mock
from typing import Generator, Any

import pytest

from app.utils.git_utils import clone_or_update_repo, clone_multiple_repos, _is_github_url, _get_github_auth_url, _add_token_to_url

@pytest.fixture
def repo_url() -> str:
    """Fixture pour l'URL du dépôt de test."""
    return "https://github.com/test/repo.git"

@pytest.fixture
def repo_dir() -> Generator[str, Any, None]:
    """Fixture pour le répertoire temporaire du dépôt."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def tmp_dir() -> Generator[str, Any, None]:
    """Fixture pour créer un répertoire temporaire."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture(autouse=True)
def clean_git_env() -> Generator[None, Any, None]:
    """Nettoie et configure l'environnement Git pour les tests."""
    old_env = dict(os.environ)
    os.environ.clear()
    os.environ["SKIP_GIT_CALLS"] = "false"
    yield
    os.environ.clear()
    os.environ.update(old_env)

def test_is_github_url() -> None:
    """Teste la détection des URLs GitHub."""
    assert _is_github_url("https://github.com/test/repo.git")
    assert _is_github_url("http://github.com/test/repo")
    assert _is_github_url("https://www.github.com/test/repo.git")
    assert not _is_github_url("https://gitlab.com/test/repo.git")
    assert not _is_github_url("git@github.com:test/repo.git")  # Ne gère pas SSH
    assert not _is_github_url("https://example.com/test/repo.git")

def test_add_token_to_url() -> None:
    """Teste l'ajout de token dans l'URL."""
    url = "https://github.com/test/repo.git"
    token = "test_token"
    expected = "https://x-access-token:test_token@github.com/test/repo.git"
    assert _add_token_to_url(url, token) == expected

    # Test avec une URL invalide
    invalid_url = "invalid_url"
    assert _add_token_to_url(invalid_url, token) == invalid_url

@pytest.mark.parametrize('url', ["https://github.com/test/repo.git"])
def test_github_actions_auth(url: str) -> None:
    """Teste l'authentification via GitHub Actions."""
    with patch.dict(os.environ, {
        "GITHUB_ACTIONS": "true",
        "GITHUB_TOKEN": "gh_test_token"
    }):
        auth_url, message = _get_github_auth_url(url)
        assert "x-access-token:gh_test_token@" in auth_url
        assert "GITHUB_TOKEN" in message
        assert "GitHub Actions" in message

@pytest.mark.parametrize('url', ["https://github.com/test/repo.git"])
def test_github_app_auth(url: str) -> None:
    """Teste l'authentification via GitHub App."""
    with (
        patch.dict(os.environ, {
            "GITHUB_APP_ID": "123456",
            "GITHUB_APP_PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----"
        }),
        patch("jwt.encode", return_value="test_jwt_token")
    ):
        auth_url, message = _get_github_auth_url(url)
        assert "x-access-token:test_jwt_token@" in auth_url
        assert "GitHub App" in message

@pytest.mark.parametrize('url', ["https://github.com/test/repo.git"])
def test_github_app_auth_failure(url: str) -> None:
    """Teste le comportement en cas d'échec d'authentification GitHub App."""
    with (
        patch.dict(os.environ, {
            "GITHUB_APP_ID": "123456",
            "GITHUB_APP_PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----"
        }),
        patch("jwt.encode", side_effect=ImportError("No module named 'jwt'"))
    ):
        auth_url, message = _get_github_auth_url(url)
        assert auth_url == url
        assert message is None

@pytest.mark.parametrize('url', ["https://github.com/test/repo.git"])
def test_github_token_auth(url: str) -> None:
    """Teste l'authentification via Personal Access Token."""
    with patch.dict(os.environ, {"GITHUB_TOKEN": "ghp_test_token"}):
        auth_url, message = _get_github_auth_url(url)
        assert "x-access-token:ghp_test_token@" in auth_url
        assert "GITHUB_TOKEN" in message

@pytest.mark.parametrize('url', ["https://github.com/test/repo.git"])
def test_no_auth(url: str) -> None:
    """Teste le comportement sans authentification."""
    auth_url, message = _get_github_auth_url(url)
    assert auth_url == url
    assert message is None

@pytest.mark.parametrize('test_data,tmp_dir', [
    ({
        'url': "https://github.com/test/repo.git",
        'token': "ghp_test_token",
        'env': {"GITHUB_TOKEN": "ghp_test_token"}
    }, None)
], indirect=['tmp_dir'])
def test_clone_with_auth(test_data: dict, tmp_dir: str) -> None:
    """Teste le clonage avec authentification."""
    with (
        patch.dict(os.environ, test_data['env']),
        patch("subprocess.run") as mock_run
    ):
        expected_auth_url = f"https://x-access-token:{test_data['token']}@github.com/test/repo.git"
        result = clone_or_update_repo(test_data['url'], tmp_dir)

        assert os.path.exists(result)
        mock_run.assert_called_once_with(
            ["git", "clone", expected_auth_url, tmp_dir],
            check=True
        )

@pytest.mark.parametrize('test_data,tmp_dir', [
    ({
        'url': "https://github.com/test/repo.git",
        'token': "ghp_test_token",
        'env': {"GITHUB_TOKEN": "ghp_test_token"}
    }, None)
], indirect=['tmp_dir'])
def test_update_with_auth(test_data: dict, tmp_dir: str) -> None:
    """Teste la mise à jour avec authentification."""
    # Simuler un repo existant
    os.makedirs(os.path.join(tmp_dir, ".git"))

    with (
        patch.dict(os.environ, test_data['env']),
        patch("subprocess.run") as mock_run
    ):
        expected_auth_url = f"https://x-access-token:{test_data['token']}@github.com/test/repo.git"
        result = clone_or_update_repo(test_data['url'], tmp_dir)

        assert result == tmp_dir
        mock_run.assert_has_calls([
            call(["git", "-C", tmp_dir, "remote", "set-url", "origin", expected_auth_url], check=True),
            call(["git", "-C", tmp_dir, "pull"], check=True)
        ])

def test_clone_multiple_repos_success() -> None:
    """Teste le clonage réussi de plusieurs dépôts."""
    repo_urls = [
        "https://github.com/test/repo1.git",
        "https://github.com/test/repo2.git",
    ]

    with (
        patch('tempfile.mkdtemp', return_value="/tmp/test_repos"),
        patch('app.utils.git_utils.clone_or_update_repo') as mock_clone
    ):
        # Configure le mock pour retourner des chemins différents pour chaque repo
        mock_clone.side_effect = [
            "/tmp/test_repos/repo1",
            "/tmp/test_repos/repo2"
        ]

        result = clone_multiple_repos(repo_urls)

        assert len(result) == 2
        assert result == ["/tmp/test_repos/repo1", "/tmp/test_repos/repo2"]

        # Vérifie que clone_or_update_repo a été appelé pour chaque URL
        expected_calls = [
            call(repo_urls[0], "/tmp/test_repos/repo1"),
            call(repo_urls[1], "/tmp/test_repos/repo2")
        ]
        mock_clone.assert_has_calls(expected_calls)

def test_clone_multiple_repos_with_error() -> None:
    """Teste le comportement lorsqu'un des clonages échoue."""
    repo_urls = [
        "https://github.com/test/repo1.git",
        "https://github.com/test/invalid-repo.git",
        "https://github.com/test/repo3.git"
    ]

    with (
        patch('tempfile.mkdtemp', return_value="/tmp/test_repos"),
        patch('app.utils.git_utils.clone_or_update_repo') as mock_clone,
        patch('app.utils.git_utils.logger') as mock_logger
    ):
        # Configure le mock pour simuler une réussite, un échec, puis une réussite
        mock_clone.side_effect = [
            "/tmp/test_repos/repo1",
            Exception("Erreur de clonage"),
            "/tmp/test_repos/repo3"
        ]

        result = clone_multiple_repos(repo_urls)

        assert len(result) == 2  # Seuls les clonages réussis sont retournés
        assert result == ["/tmp/test_repos/repo1", "/tmp/test_repos/repo3"]

        # Vérifie que l'erreur a été journalisée
        mock_logger.error.assert_called_once()
        error_message = mock_logger.error.call_args[0][0]
        assert "Erreur lors du clonage du dépôt" in error_message

def test_clone_multiple_repos_empty_list() -> None:
    """Teste le comportement avec une liste vide d'URLs."""
    with patch('tempfile.mkdtemp', return_value="/tmp/test_repos"):
        result = clone_multiple_repos([])
        assert result == []

def test_clone_multiple_repos_with_non_git_urls() -> None:
    """Teste le clonage avec des URLs non-git."""
    repo_urls = [
        "https://github.com/test/repo1",  # Sans .git
        "https://example.com/repo2",      # Non GitHub
    ]

    with (
        patch('tempfile.mkdtemp', return_value="/tmp/test_repos"),
        patch('app.utils.git_utils.clone_or_update_repo') as mock_clone
    ):
        mock_clone.side_effect = [
            "/tmp/test_repos/repo_0",
            "/tmp/test_repos/repo_1"
        ]

        result = clone_multiple_repos(repo_urls)

        assert len(result) == 2
        # Vérifie que les noms de dossiers par défaut sont utilisés
        expected_calls = [
            call(repo_urls[0], "/tmp/test_repos/repo_0"),
            call(repo_urls[1], "/tmp/test_repos/repo_1")
        ]
        mock_clone.assert_has_calls(expected_calls)