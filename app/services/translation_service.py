"""Service de gestion de la traduction avec chargement paresseux.

Ce service implémente le pattern singleton et assure
que le modèle de traduction est chargé une seule fois.
"""

import logging

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


class TranslationService:
    """Service de gestion des modèles de traduction."""

    _instance = None
    _model = None
    _tokenizer = None

    @classmethod
    def get_instance(cls) -> "TranslationService":
        """Récupère l'instance singleton du service.

        Returns:
            TranslationService: Instance unique du service
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, model_name: str = "facebook/nllb-200-distilled-600M") -> None:
        """Charge le modèle et le tokenizer de traduction.

        Args:
            model_name: Nom du modèle à charger
        """
        try:
            logger.info("Chargement du modèle de traduction %s", model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            logger.info("Modèle de traduction chargé avec succès")
        except Exception as e:
            logger.error("Erreur lors du chargement du modèle de traduction: %s", e)
            raise

    @property
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé.

        Returns:
            bool: True si le modèle est chargé, False sinon
        """
        return self._model is not None and self._tokenizer is not None

    @property
    def model_and_tokenizer(self) -> tuple[AutoTokenizer, AutoModelForSeq2SeqLM]:
        """Récupère le modèle et le tokenizer.

        Returns:
            Tuple: (tokenizer, model)

        Raises:
            RuntimeError: Si le modèle n'est pas chargé
        """
        if not self.is_loaded:
            raise RuntimeError("Le modèle de traduction n'est pas chargé")
        return self._tokenizer, self._model
