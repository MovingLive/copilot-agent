"""
Tests unitaires pour le module export_utils.
"""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch

import boto3
import pytest
from botocore.exceptions import ClientError
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
    """Crée un répertoire source temporaire pour les tests."""
    temp_dir = tempfile.mkdtemp()
    # Créer quelques fichiers de test
    with open(os.path.join(temp_dir, "test1.txt"), "w", encoding="utf-8") as f:
        f.write("Contenu de test 1")

    # Créer un sous-répertoire
    subdir = os.path.join(temp_dir, "subdir")
    os.makedirs(subdir)
    with open(os.path.join(subdir, "test2.txt"), "w", encoding="utf-8") as f:
        f.write("Contenu de test 2")

    yield temp_dir
    # Nettoyage
    shutil.rmtree(temp_dir)


@pytest.fixture
def temp_dest_dir():
    """Crée un répertoire de destination temporaire pour les tests."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Nettoyage
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_s3_bucket():
    """Configure un bucket S3 simulé pour les tests."""
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")
        yield s3


@pytest.fixture(autouse=True)
def mock_env():
    """Fixture pour s'assurer que les variables d'environnement sont correctement définies."""
    with patch.dict(os.environ, {"ENV": "local"}):
        yield


# --- Tests ---
def test_is_local_environment():
    """Teste la fonction is_local_environment."""
    # Test du cas où TESTING n'est pas activé
    with patch("app.core.config.settings.ENV", "local"):
        with patch("app.core.config.settings.TESTING", False):
            assert is_local_environment() is True

    with patch("app.core.config.settings.ENV", "production"):
        with patch("app.core.config.settings.TESTING", False):
            assert is_local_environment() is False

    # Test du cas où TESTING est activé
    with patch("app.core.config.settings.ENV", "local"):
        with patch("app.core.config.settings.TESTING", True):
            assert is_local_environment() is False


def test_copy_to_local_output(temp_source_dir, temp_dest_dir):
    """Teste la fonction copy_to_local_output."""
    with patch("app.core.config.settings.LOCAL_OUTPUT_DIR", temp_dest_dir):
        # Tester avec le répertoire de destination par défaut
        copy_to_local_output(temp_source_dir)

        # Vérifier que les fichiers ont été copiés
        assert os.path.exists(os.path.join(temp_dest_dir, "test1.txt"))
        assert os.path.exists(os.path.join(temp_dest_dir, "subdir", "test2.txt"))

        # Vérifier le contenu
        with open(os.path.join(temp_dest_dir, "test1.txt"), encoding="utf-8") as f:
            assert f.read() == "Contenu de test 1"
        with open(
            os.path.join(temp_dest_dir, "subdir", "test2.txt"), encoding="utf-8"
        ) as f:
            assert f.read() == "Contenu de test 2"


def test_copy_to_local_output_custom_dir(temp_source_dir, temp_dest_dir):
    """Teste la fonction copy_to_local_output avec un répertoire personnalisé."""
    # Tester avec un répertoire de destination personnalisé
    custom_dir = os.path.join(temp_dest_dir, "custom")
    copy_to_local_output(temp_source_dir, custom_dir)

    # Vérifier que les fichiers ont été copiés
    assert os.path.exists(os.path.join(custom_dir, "test1.txt"))
    assert os.path.exists(os.path.join(custom_dir, "subdir", "test2.txt"))


def test_upload_directory_to_s3(temp_source_dir, mock_s3_bucket):
    """Teste la fonction upload_directory_to_s3."""
    bucket_name = "test-bucket"
    prefix = "test-prefix"

    with patch("app.core.config.settings.AWS_REGION", "us-east-1"):
        with patch("app.core.config.settings.AWS_ACCESS_KEY_ID", "test"):
            with patch("app.core.config.settings.AWS_SECRET_ACCESS_KEY", "test"):
                # Upload des fichiers
                upload_directory_to_s3(temp_source_dir, bucket_name, prefix)

    # Vérifier que les fichiers ont été uploadés
    s3 = boto3.client("s3", region_name="us-east-1")
    objects = s3.list_objects(Bucket=bucket_name, Prefix=prefix)
    assert "Contents" in objects
    assert len(objects["Contents"]) == 2

    # Vérifier les chemins
    keys = [obj["Key"] for obj in objects["Contents"]]
    assert f"{prefix}/test1.txt" in keys
    assert f"{prefix}/subdir/test2.txt" in keys


def test_upload_directory_to_s3_error(temp_source_dir):
    """Teste la gestion des erreurs dans upload_directory_to_s3."""
    with patch("boto3.client") as mock_boto3:
        mock_client = MagicMock()
        mock_boto3.return_value = mock_client
        mock_client.upload_file.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Access Denied"}},
            "upload_file",
        )

        with patch("app.core.config.settings.AWS_REGION", "us-east-1"):
            with patch("app.core.config.settings.AWS_ACCESS_KEY_ID", "test"):
                with patch("app.core.config.settings.AWS_SECRET_ACCESS_KEY", "test"):
                    with patch(
                        "app.core.config.settings.S3_BUCKET_NAME", "test-bucket"
                    ):
                        # L'upload devrait échouer sans lever d'exception
                        upload_directory_to_s3(temp_source_dir)


def test_export_data_local(temp_source_dir, temp_dest_dir):
    """Teste la fonction export_data en environnement local."""
    with patch("app.utils.export_utils.is_local_environment", return_value=True):
        with patch("app.core.config.settings.LOCAL_OUTPUT_DIR", temp_dest_dir):
            # Exporter les données
            export_data(temp_source_dir)

    # Vérifier que les fichiers ont été copiés localement
    assert os.path.exists(os.path.join(temp_dest_dir, "test1.txt"))
    assert os.path.exists(os.path.join(temp_dest_dir, "subdir", "test2.txt"))


def test_export_data_s3(temp_source_dir, mock_s3_bucket):
    """Teste la fonction export_data en environnement non-local (S3)."""
    bucket_name = "test-bucket"
    prefix = "test-prefix"

    with patch("app.utils.export_utils.is_local_environment", return_value=False):
        with patch("app.core.config.settings.S3_BUCKET_NAME", bucket_name):
            with patch("app.core.config.settings.AWS_REGION", "us-east-1"):
                with patch("app.core.config.settings.AWS_ACCESS_KEY_ID", "test"):
                    with patch(
                        "app.core.config.settings.AWS_SECRET_ACCESS_KEY", "test"
                    ):
                        # Exporter les données
                        export_data(temp_source_dir, prefix, bucket_name)

    # Vérifier que les fichiers ont été uploadés sur S3
    s3 = boto3.client("s3", region_name="us-east-1")
    objects = s3.list_objects(Bucket=bucket_name, Prefix=prefix)
    assert "Contents" in objects
    assert len(objects["Contents"]) == 2
