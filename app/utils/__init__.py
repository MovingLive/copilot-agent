"""Package d'utilitaires pour l'application.
Centralise les fonctionnalités communes utilisées par différentes parties de l'application.
"""

from app.utils.export_utils import (
    copy_to_local_output,
    export_data,
    is_local_environment,
    upload_directory_to_s3,
)
from app.utils.git_utils import clone_or_update_repo
from app.utils.markdown_utils import read_markdown_files, segment_text
from app.utils.vector_db_utils import (
    process_documents_for_chroma,
    process_documents_for_faiss,
)

__all__ = [
    "clone_or_update_repo",
    "read_markdown_files",
    "segment_text",
    "process_documents_for_chroma",
    "process_documents_for_faiss",
    "is_local_environment",
    "copy_to_local_output",
    "upload_directory_to_s3",
    "export_data",
]
