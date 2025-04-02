"""
Tests unitaires pour le module export_utils.
"""

import os
import tempfile
from unittest.mock import patch
from datetime import datetime, timezone

import boto3
import pytest
from moto import mock_aws

from app.utils.export_utils import (
    copy_to_local_output,
    export_data,
    is_local_environment,
    upload_directory_to_s3,
)


# --- Fixtures ---
@pytest.fixture
def temp_source_dir():
    """Crée un répertoire source temporaire avec des fichiers de test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Créer quelques fichiers de test
        file1_path = os.path.join(temp_dir, "test1.txt")
        file2_path = os.path.join(temp_dir, "test2.txt")

        with open(file1_path, "w") as f:
            f.write("Contenu test 1")
        with open(file2_path, "w") as f:
            f.write("Contenu test 2")

        # Créer un sous-répertoire avec un fichier
        subdir = os.path.join(temp_dir, "subdir")
        os.makedirs(subdir)
        subfile_path = os.path.join(subdir, "test3.txt")
        with open(subfile_path, "w") as f:
            f.write("Contenu test 3")

        yield temp_dir


@pytest.fixture
def temp_dest_dir():
    """Crée un répertoire de destination temporaire."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_s3_bucket():
    """Configure un mock de bucket S3 pour les tests."""
    with mock_aws():
        conn = boto3.client(
            "s3",
            region_name="ca-central-1",
            aws_access_key_id="testing",
            aws_secret_access_key="testing",
        )
        conn.create_bucket(
            Bucket="test-bucket",
            CreateBucketConfiguration={"LocationConstraint": "ca-central-1"},
        )
        yield conn


@pytest.fixture(autouse=True)
def mock_env():
    """Fixture pour s'assurer que les variables d'environnement sont correctement définies."""
    with patch.dict(os.environ, {"ENV": "local"}):
        yield


# --- Tests ---
def test_is_local_environment():
    """Teste la fonction is_local_environment."""
    # Test du cas où TESTING n'est pas activé
    with patch.dict(os.environ, {"ENV": "local", "TESTING": "false"}):
        assert is_local_environment() is True

    with patch.dict(os.environ, {"ENV": "production", "TESTING": "false"}):
        assert is_local_environment() is False

    # Test du cas où TESTING est activé (comme dans GitHub Actions)
    with patch.dict(os.environ, {"ENV": "local", "TESTING": "true"}):
        assert is_local_environment() is False


def test_copy_to_local_output(temp_source_dir, temp_dest_dir):
    """Teste la fonction copy_to_local_output."""
    # Test avec destination personnalisée
    copy_to_local_output(temp_source_dir, temp_dest_dir)

    # Vérifier que tous les fichiers ont été copiés
    assert os.path.exists(os.path.join(temp_dest_dir, "test1.txt"))
    assert os.path.exists(os.path.join(temp_dest_dir, "test2.txt"))
    assert os.path.exists(os.path.join(temp_dest_dir, "subdir", "test3.txt"))

    # Vérifier le contenu d'un fichier
    with open(os.path.join(temp_dest_dir, "test1.txt"), "r") as f:
        assert f.read() == "Contenu test 1"


def test_upload_directory_to_s3(temp_source_dir, mock_s3_bucket):
    """Teste la fonction upload_directory_to_s3."""
    bucket_name = "test-bucket"
    prefix = "test-prefix"

    # Upload des fichiers
    upload_directory_to_s3(temp_source_dir, bucket_name, prefix)

    # Vérifier que les fichiers ont été uploadés
    s3_objects = mock_s3_bucket.list_objects(Bucket=bucket_name, Prefix=prefix)
    uploaded_files = [obj["Key"] for obj in s3_objects["Contents"]]

    assert f"{prefix}/test1.txt" in uploaded_files
    assert f"{prefix}/test2.txt" in uploaded_files
    assert f"{prefix}/subdir/test3.txt" in uploaded_files


def test_export_data_local(temp_source_dir, temp_dest_dir):
    """Teste la fonction export_data en environnement local."""
    with (
        patch("app.utils.export_utils.is_local_environment", return_value=True),
        patch("app.utils.export_utils.LOCAL_OUTPUT_DIR", temp_dest_dir),
    ):
        export_data(temp_source_dir)

        # Vérifier que les fichiers ont été copiés localement
        assert os.path.exists(os.path.join(temp_dest_dir, "test1.txt"))
        assert os.path.exists(os.path.join(temp_dest_dir, "test2.txt"))
        assert os.path.exists(os.path.join(temp_dest_dir, "subdir", "test3.txt"))


def test_export_data_s3(temp_source_dir, mock_s3_bucket):
    """Teste la fonction export_data en environnement non-local (S3)."""
    bucket_name = "test-bucket"
    prefix = "test-prefix"

    with (
        patch("app.utils.export_utils.is_local_environment", return_value=False),
        patch("app.utils.export_utils.S3_BUCKET_NAME", bucket_name),
    ):
        export_data(temp_source_dir, prefix)

        # Vérifier que les fichiers ont été uploadés sur S3
        s3_objects = mock_s3_bucket.list_objects(Bucket=bucket_name, Prefix=prefix)
        uploaded_files = [obj["Key"] for obj in s3_objects["Contents"]]

        assert f"{prefix}/test1.txt" in uploaded_files
        assert f"{prefix}/test2.txt" in uploaded_files
        assert f"{prefix}/subdir/test3.txt" in uploaded_files
