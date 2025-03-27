"""Tests unitaires pour le module git_utils."""

import os
import tempfile
from unittest.mock import patch, call
from typing import Generator, Any

import pytest

from app.utils.git_utils import clone_or_update_repo, _is_github_url, _get_github_auth_url, _add_token_to_url

@pytest.fixture
def sample_repo_url() -> str:
    """Fixture pour l'URL du dépôt de test."""
    return "https://github.com/test/repo.git"

@pytest.fixture
def sample_repo_dir() -> Generator[str, Any, None]:
    """Fixture pour le répertoire temporaire du dépôt."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture(autouse=True)
def disable_skip_git_calls() -> Generator[None, Any, None]:
    """Désactiver temporairement la variable SKIP_GIT_CALLS."""
    old_value = os.environ.get("SKIP_GIT_CALLS")
    os.environ["SKIP_GIT_CALLS"] = "false"
    yield
    if old_value is not None:
        os.environ["SKIP_GIT_CALLS"] = old_value
    else:
        del os.environ["SKIP_GIT_CALLS"]

@pytest.fixture(autouse=True)
def clear_env() -> Generator[None, Any, None]:
    """Nettoie l'environnement avant chaque test."""
    old_env = dict(os.environ)
    os.environ.clear()
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

def test_github_actions_auth(sample_repo_url) -> None:
    """Teste l'authentification via GitHub Actions."""
    with patch.dict(os.environ, {
        "GITHUB_ACTIONS": "true",
        "GITHUB_TOKEN": "gh_test_token"
    }):
        auth_url, message = _get_github_auth_url(sample_repo_url)
        assert "x-access-token:gh_test_token@" in auth_url
        assert "GITHUB_TOKEN" in message
        assert "GitHub Actions" in message

def test_github_app_auth(sample_repo_url) -> None:
    """Teste l'authentification via GitHub App."""
    with (
        patch.dict(os.environ, {
            "GITHUB_APP_ID": "123456",
            "GITHUB_APP_PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----"
        }),
        patch("jwt.encode", return_value="test_jwt_token")
    ):
        auth_url, message = _get_github_auth_url(sample_repo_url)
        assert "x-access-token:test_jwt_token@" in auth_url
        assert "GitHub App" in message

def test_github_app_auth_failure(sample_repo_url) -> None:
    """Teste le comportement en cas d'échec d'authentification GitHub App."""
    with (
        patch.dict(os.environ, {
            "GITHUB_APP_ID": "123456",
            "GITHUB_APP_PRIVATE_KEY": "-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----"
        }),
        patch("jwt.encode", side_effect=ImportError("No module named 'jwt'"))
    ):
        auth_url, message = _get_github_auth_url(sample_repo_url)
        assert auth_url == sample_repo_url
        assert message is None

def test_github_pat_auth(sample_repo_url) -> None:
    """Teste l'authentification via Personal Access Token."""
    with patch.dict(os.environ, {"GITHUB_PAT": "ghp_test_pat"}):
        auth_url, message = _get_github_auth_url(sample_repo_url)
        assert "x-access-token:ghp_test_pat@" in auth_url
        assert "Personal Access Token" in message

def test_no_auth(sample_repo_url) -> None:
    """Teste le comportement sans authentification."""
    auth_url, message = _get_github_auth_url(sample_repo_url)
    assert auth_url == sample_repo_url
    assert message is None

def test_clone_with_pat(sample_repo_url, sample_repo_dir) -> None:
    """Teste le clonage avec authentification PAT."""
    with (
        patch.dict(os.environ, {"GITHUB_PAT": "ghp_test_pat"}),
        patch("subprocess.run") as mock_run
    ):
        expected_auth_url = "https://x-access-token:ghp_test_pat@github.com/test/repo.git"
        result = clone_or_update_repo(sample_repo_url, sample_repo_dir)
        
        assert os.path.exists(result)
        mock_run.assert_called_once_with(
            ["git", "clone", expected_auth_url, sample_repo_dir],
            check=True
        )

def test_update_with_pat(sample_repo_url, sample_repo_dir) -> None:
    """Teste la mise à jour avec authentification PAT."""
    # Simuler un repo existant
    os.makedirs(os.path.join(sample_repo_dir, ".git"))
    
    with (
        patch.dict(os.environ, {"GITHUB_PAT": "ghp_test_pat"}),
        patch("subprocess.run") as mock_run
    ):
        expected_auth_url = "https://x-access-token:ghp_test_pat@github.com/test/repo.git"
        result = clone_or_update_repo(sample_repo_url, sample_repo_dir)
        
        assert result == sample_repo_dir
        mock_run.assert_has_calls([
            call(["git", "-C", sample_repo_dir, "remote", "set-url", "origin", expected_auth_url], check=True),
            call(["git", "-C", sample_repo_dir, "pull"], check=True)
        ])