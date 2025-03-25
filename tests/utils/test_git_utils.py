"""Tests unitaires pour le module git_utils."""

import os
import tempfile
from unittest.mock import patch

import pytest

from app.utils.git_utils import clone_or_update_repo

@pytest.fixture
def sample_repo_url() -> str:
    return "https://github.com/test/repo.git"

@pytest.fixture
def sample_repo_dir() -> str:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir

@pytest.fixture
def disable_skip_git_calls():
    """Désactiver temporairement la variable SKIP_GIT_CALLS."""
    old_value = os.environ.get("SKIP_GIT_CALLS")
    os.environ["SKIP_GIT_CALLS"] = "false"
    yield
    if old_value is not None:
        os.environ["SKIP_GIT_CALLS"] = old_value
    else:
        del os.environ["SKIP_GIT_CALLS"]

def test_clone_repo_new_directory(
    sample_repo_url: str, sample_repo_dir: str, disable_skip_git_calls
) -> None:
    with patch("subprocess.run") as mock_run:
        result = clone_or_update_repo(sample_repo_url, sample_repo_dir)
        assert os.path.exists(result)
        mock_run.assert_called_once_with(
            ["git", "clone", sample_repo_url, sample_repo_dir], check=True
        )

def test_update_existing_repo(
    sample_repo_url: str, sample_repo_dir: str, disable_skip_git_calls
) -> None:
    # Simuler un repo existant
    os.makedirs(os.path.join(sample_repo_dir, ".git"))

    with patch("subprocess.run") as mock_run:
        result = clone_or_update_repo(sample_repo_url, sample_repo_dir)
        assert result == sample_repo_dir
        mock_run.assert_called_once_with(
            ["git", "-C", sample_repo_dir, "pull"], check=True
        )