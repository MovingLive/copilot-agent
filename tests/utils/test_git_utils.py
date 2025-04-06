"""Tests pour les utilitaires git."""

import os
import tempfile
from unittest.mock import call, patch

import pytest

from app.utils.git_utils import (
    _add_token_to_url,
    _get_github_auth_url,
    _is_github_url,
    clone_multiple_repos,
    clone_or_update_repo,
)


# --- Tests pour les fonctions internes ---
@pytest.mark.parametrize(
    "url,expected",
    [
        ("https://github.com/test/repo.git", True),
        ("http://github.com/test/repo", True),
        ("https://www.github.com/test/repo", True),
        ("https://gitlab.com/test/repo", False),
        ("git@github.com:test/repo.git", False),  # Format SSH non supporté actuellement
    ],
)
def test_is_github_url(url: str, expected: bool) -> None:
    """Teste la fonction _is_github_url."""
    assert _is_github_url(url) is expected


@pytest.mark.parametrize("url", ["https://github.com/test/repo.git"])
def test_github_token_auth(url: str) -> None:
    """Teste l'authentification via Personal Access Token."""
    with patch("app.core.config.settings.GITHUB_TOKEN", "ghp_test_token"):
        auth_url, message = _get_github_auth_url(url)
        assert "x-access-token:ghp_test_token@" in auth_url


@pytest.mark.parametrize("url", ["https://github.com/test/repo.git"])
def test_github_app_auth(url: str) -> None:
    """Teste l'authentification via GitHub App."""
    with (
        patch("app.core.config.settings.GITHUB_TOKEN", ""),
        patch("app.core.config.settings.GITHUB_APP_ID", "123456"),
        patch(
            "app.core.config.settings.GITHUB_APP_PRIVATE_KEY",
            "-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----",
        ),
        patch("jwt.encode", return_value="test_jwt_token"),
    ):
        auth_url, message = _get_github_auth_url(url)
        assert "x-access-token:test_jwt_token@" in auth_url


@pytest.mark.parametrize("url", ["https://github.com/test/repo.git"])
def test_github_app_auth_failure(url: str) -> None:
    """Teste le comportement en cas d'échec d'authentification GitHub App."""
    with (
        patch("app.core.config.settings.GITHUB_TOKEN", ""),
        patch("app.core.config.settings.GITHUB_APP_ID", "123456"),
        patch(
            "app.core.config.settings.GITHUB_APP_PRIVATE_KEY",
            "-----BEGIN RSA PRIVATE KEY-----\ntest\n-----END RSA PRIVATE KEY-----",
        ),
        patch("jwt.encode", side_effect=ImportError("No module named 'jwt'")),
    ):
        auth_url, message = _get_github_auth_url(url)
        assert auth_url == url


@pytest.mark.parametrize("url", ["https://github.com/test/repo.git"])
def test_github_actions_auth(url: str) -> None:
    """Teste l'authentification via GitHub Actions."""
    with patch("app.core.config.settings.GITHUB_TOKEN", "gh_test_token"):
        auth_url, message = _get_github_auth_url(url)
        assert "x-access-token:gh_test_token@" in auth_url


@pytest.mark.parametrize("url", ["https://github.com/test/repo.git"])
def test_no_auth(url: str) -> None:
    """Teste le comportement sans authentification."""
    with patch("app.core.config.settings.GITHUB_TOKEN", ""):
        auth_url, message = _get_github_auth_url(url)
        assert auth_url == url


def test_add_token_to_url() -> None:
    """Teste la fonction _add_token_to_url."""
    url = "https://github.com/test/repo.git"
    token = "test_token"
    expected = "https://x-access-token:test_token@github.com/test/repo.git"
    assert _add_token_to_url(url, token) == expected

    # Test avec une URL non-HTTP
    url = "git@github.com:test/repo.git"
    assert _add_token_to_url(url, token) == url


# --- Tests pour les fonctions principales ---
@pytest.fixture
def tmp_dir():
    """Fixture pour créer un répertoire temporaire."""
    tmpdir = tempfile.mkdtemp()
    yield tmpdir
    # Cleanup handled by the OS's temp file system


@pytest.mark.parametrize(
    "test_data,tmp_dir",
    [
        (
            {
                "url": "https://github.com/test/repo.git",
                "token": "ghp_test_token",
                "env": {"GITHUB_TOKEN": "ghp_test_token"},
            },
            None,
        )
    ],
    indirect=["tmp_dir"],
)
def test_clone_with_auth(test_data: dict, tmp_dir: str) -> None:
    """Teste le clonage avec authentification."""
    with (
        patch("app.core.config.settings.GITHUB_TOKEN", test_data["token"]),
        patch("subprocess.run") as mock_run,
    ):
        expected_auth_url = (
            f"https://x-access-token:{test_data['token']}@github.com/test/repo.git"
        )
        result = clone_or_update_repo(test_data["url"], tmp_dir)

        assert os.path.exists(result)
        mock_run.assert_called_once_with(
            ["git", "clone", expected_auth_url, tmp_dir], check=True
        )


@pytest.mark.parametrize(
    "test_data,tmp_dir",
    [
        (
            {
                "url": "https://github.com/test/repo.git",
                "token": "ghp_test_token",
                "env": {"GITHUB_TOKEN": "ghp_test_token"},
            },
            None,
        )
    ],
    indirect=["tmp_dir"],
)
def test_update_with_auth(test_data: dict, tmp_dir: str) -> None:
    """Teste la mise à jour avec authentification."""
    # Simuler un repo existant
    os.makedirs(os.path.join(tmp_dir, ".git"))

    with (
        patch("app.core.config.settings.GITHUB_TOKEN", test_data["token"]),
        patch("subprocess.run") as mock_run,
    ):
        expected_auth_url = (
            f"https://x-access-token:{test_data['token']}@github.com/test/repo.git"
        )
        result = clone_or_update_repo(test_data["url"], tmp_dir)

        assert result == tmp_dir
        mock_run.assert_has_calls(
            [
                call(
                    [
                        "git",
                        "-C",
                        tmp_dir,
                        "remote",
                        "set-url",
                        "origin",
                        expected_auth_url,
                    ],
                    check=True,
                ),
                call(["git", "-C", tmp_dir, "pull"], check=True),
            ]
        )


def test_clone_multiple_repos():
    """Teste la fonction clone_multiple_repos."""
    repo_urls = [
        "https://github.com/test/repo1.git",
        "https://github.com/test/repo2.git",
    ]

    with patch("app.utils.git_utils.clone_or_update_repo") as mock_clone:
        mock_clone.side_effect = lambda url, path: path
        result = clone_multiple_repos(repo_urls)

        assert len(result) == 2
        assert all(os.path.basename(path) in ["repo1", "repo2"] for path in result)
        assert mock_clone.call_count == 2


def test_clone_multiple_repos_error():
    """Teste la fonction clone_multiple_repos en cas d'erreur."""
    repo_urls = [
        "https://github.com/test/repo1.git",
        "https://github.com/test/repo2.git",
    ]

    with patch("app.utils.git_utils.clone_or_update_repo") as mock_clone:
        # Le premier clonage réussit, le deuxième échoue
        mock_clone.side_effect = [
            os.path.join(tempfile.gettempdir(), "repo1"),
            Exception("Test error"),
        ]
        result = clone_multiple_repos(repo_urls)

        assert len(result) == 1
        assert os.path.basename(result[0]) == "repo1"
        assert mock_clone.call_count == 2
