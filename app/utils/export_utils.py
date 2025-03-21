"""
Utilitaires pour la gestion des exportations de fichiers.
Ce module centralise les fonctions d'exportation locale ou vers S3.
"""

import logging
import os
import shutil

import boto3
from dotenv import load_dotenv

# --- Chargement des variables d'environnement ---
load_dotenv()

# --- Configuration ---
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "mon-bucket-faiss")
AWS_REGION = os.getenv("AWS_REGION", "ca-central-1")

# Répertoire de sortie local (utilisé si ENV = "local")
LOCAL_OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "output",
)

# --- Logger ---
logger = logging.getLogger("export_utils")


def is_local_environment() -> bool:
    """
    Vérifie si l'environnement courant est local.

    Returns:
        bool: True si l'environnement est local, False sinon
    """
    return os.getenv("ENV", "local").strip().lower() == "local"


def copy_to_local_output(source_dir: str, destination_dir: str = None) -> None:
    """
    Copie les fichiers du répertoire source vers le répertoire de sortie local.

    Args:
        source_dir (str): Répertoire source
        destination_dir (str, optional): Répertoire de destination personnalisé
                                       Si None, utilise LOCAL_OUTPUT_DIR
    """
    if destination_dir is None:
        destination_dir = LOCAL_OUTPUT_DIR

    if not os.path.exists(destination_dir):
        os.makedirs(destination_dir)

    # Vider d'abord le répertoire de destination
    for item in os.listdir(destination_dir):
        item_path = os.path.join(destination_dir, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        elif os.path.isdir(item_path):
            shutil.rmtree(item_path)

    # Copier les fichiers
    for root, _, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        destination_path = os.path.join(destination_dir, relative_path)

        if not os.path.exists(destination_path) and relative_path != ".":
            os.makedirs(destination_path)

        for file in files:
            source_file_path = os.path.join(root, file)
            dest_file_path = os.path.join(destination_path, file)
            logger.info("Copie de %s vers %s...", source_file_path, dest_file_path)
            try:
                shutil.copy2(source_file_path, dest_file_path)
            except Exception as e:
                logger.error("Erreur lors de la copie de %s: %s", source_file_path, e)


def upload_directory_to_s3(
    directory: str, bucket_name: str = None, prefix: str = ""
) -> None:
    """
    Upload l'intégralité des fichiers du répertoire 'directory' vers le bucket S3 sous le préfixe 'prefix'.

    Args:
        directory (str): Répertoire à uploader
        bucket_name (str, optional): Nom du bucket S3. Si None, utilise S3_BUCKET_NAME
        prefix (str): Préfixe pour les fichiers dans le bucket
    """
    if bucket_name is None:
        bucket_name = S3_BUCKET_NAME

    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID", "testing"),
        aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY", "testing"),
    )

    for root, _, files in os.walk(directory):
        for file in files:
            full_path = os.path.join(root, file)
            relative_path = os.path.relpath(full_path, directory)
            s3_key = os.path.join(prefix, relative_path).replace("\\", "/")
            logger.info(
                "Téléversement de %s vers s3://%s/%s...", full_path, bucket_name, s3_key
            )
            try:
                s3_client.upload_file(full_path, bucket_name, s3_key)
            except Exception as e:
                logger.error("Erreur lors du téléversement de %s: %s", full_path, e)


def export_data(source_dir: str, s3_prefix: str = "", bucket_name: str = None) -> None:
    """
    Exporte les données soit vers un répertoire local soit vers S3 selon l'environnement.

    Args:
        source_dir (str): Répertoire source contenant les données à exporter
        s3_prefix (str): Préfixe à utiliser dans le bucket S3 si exportation vers S3
        bucket_name (str, optional): Nom du bucket S3. Si None, utilise S3_BUCKET_NAME
    """
    if is_local_environment():
        logger.info(
            "Environnement local détecté. Copie vers le répertoire local %s...",
            LOCAL_OUTPUT_DIR,
        )
        copy_to_local_output(source_dir)
        logger.info("Copie terminée.")
    else:
        logger.info("Début de la synchronisation du dossier vers S3...")
        upload_directory_to_s3(source_dir, bucket_name, s3_prefix)
        logger.info("Synchronisation terminée.")
