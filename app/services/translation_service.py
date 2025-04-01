"""Service de gestion de la traduction avec deep-translator.

Ce service implémente le pattern singleton pour la gestion des traductions.
"""

import logging

from deep_translator import GoogleTranslator

logger = logging.getLogger(__name__)


class TranslationService:
    """Service de gestion des traductions."""

    _instance = None
    _translator: GoogleTranslator | None = None

    @classmethod
    def get_instance(cls) -> "TranslationService":
        """Récupère l'instance singleton du service.

        Returns:
            TranslationService: Instance unique du service
        """
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self, source_lang: str = "auto", target_lang: str = "en") -> None:
        """Initialise le traducteur.

        Args:
            source_lang: Langue source (défaut: auto-détection)
            target_lang: Langue cible (défaut: anglais)
        """
        try:
            logger.info(
                "Initialisation du traducteur %s -> %s", source_lang, target_lang
            )
            self._translator = GoogleTranslator(source=source_lang, target=target_lang)
            logger.info("Traducteur initialisé avec succès")
        except Exception as e:
            logger.error("Erreur lors de l'initialisation du traducteur: %s", e)
            self._translator = None  # Réinitialisation du translator en cas d'erreur
            raise

    @property
    def is_loaded(self) -> bool:
        """Vérifie si le traducteur est initialisé.

        Returns:
            bool: True si le traducteur est initialisé, False sinon
        """
        return self._translator is not None

    def translate(
        self, text: str, source_lang: str = "auto", target_lang: str = "en"
    ) -> str:
        """Traduit un texte.

        Args:
            text: Texte à traduire
            source_lang: Langue source (défaut: auto-détection)
            target_lang: Langue cible (défaut: anglais)

        Returns:
            str: Texte traduit
        """
        # Initialisation automatique du traducteur si nécessaire
        if not self.is_loaded or (
            self._translator
            and (
                source_lang != self._translator.source
                or target_lang != self._translator.target
            )
        ):
            self.load_model(source_lang, target_lang)

        try:
            return self._translator.translate(text)
        except Exception as e:
            logger.error("Erreur lors de la traduction: %s", e)
            return text
