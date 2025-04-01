"""Tests pour le module translation_utils."""

import pytest
from unittest.mock import patch, MagicMock

from app.utils.translation_utils import (
    detect_language,
    needs_translation,
    translate_text,
)
from app.services.translation_service import TranslationService


@pytest.fixture
def mock_translation_service():
    """Fixture pour mocker le service de traduction."""
    with patch.object(TranslationService, "get_instance") as mock_get_instance:
        service_mock = MagicMock()
        service_mock.is_loaded = True
        service_mock.translate.return_value = "Translated text"
        mock_get_instance.return_value = service_mock
        yield service_mock


class TestLanguageDetection:
    """Tests pour la détection de langue."""

    def test_detect_language_french(self):
        """Vérifier que le français est correctement détecté."""
        text = "Bonjour, comment allez-vous aujourd'hui ?"
        assert detect_language(text) == "fr"

    def test_detect_language_english(self):
        """Vérifier que l'anglais est correctement détecté."""
        text = "Hello, how are you today?"
        assert detect_language(text) == "en"

    def test_detect_language_empty(self):
        """Vérifier la gestion des textes vides."""
        with pytest.raises(Exception):
            detect_language("")

    def test_detect_language_error_handling(self):
        """Vérifier que les erreurs sont gérées correctement."""
        with patch("langid.classify", side_effect=Exception("Test error")):
            assert detect_language("Test") == "unknown"


class TestTranslationNeeded:
    """Tests pour déterminer si une traduction est nécessaire."""

    @pytest.mark.parametrize(
        "text,target_lang,expected",
        [
            ("Hello, world!", "en", False),  # Pas de traduction nécessaire
            ("Bonjour le monde !", "en", True),  # Traduction nécessaire
            ("", "en", False),  # Texte vide
            ("Hello, world!", "fr", True),  # Traduction nécessaire
        ],
    )
    def test_needs_translation(self, text, target_lang, expected):
        """Vérifier la détection correcte du besoin de traduction."""
        with patch("app.utils.translation_utils.detect_language",
                  side_effect=lambda x: "en" if "Hello" in x else "fr"):
            assert needs_translation(text, target_lang) == expected


class TestTranslation:
    """Tests pour la fonction de traduction."""

    def test_translate_text_same_language(self):
        """Vérifier que le texte n'est pas traduit si la langue est la même."""
        text = "Hello, world!"
        assert translate_text(text, src_lang="en", tgt_lang="en") == text

    def test_translate_text_different_language(self, mock_translation_service):
        """Vérifier que la traduction est effectuée correctement avec le service de traduction."""
        text = "Bonjour le monde !"
        translated = translate_text(text, src_lang="fr", tgt_lang="en")
        assert translated == "Translated text"

        # Vérifier que le service a été appelé avec les bons paramètres
        mock_translation_service.translate.assert_called_once_with(
            text, source_lang="fr", target_lang="en"
        )

    def test_translate_text_error_handling(self, mock_translation_service):
        """Vérifier la gestion des erreurs lors de la traduction."""
        mock_translation_service.translate.side_effect = Exception("Test error")

        # En cas d'erreur, le texte original doit être retourné
        original = "Bonjour le monde !"
        assert translate_text(original, src_lang="fr", tgt_lang="en") == original

    def test_translate_text_empty(self):
        """Vérifier la gestion des textes vides."""
        assert translate_text("") == ""
        assert translate_text("   ") == "   "

    def test_translate_text_unknown_language(self):
        """Vérifier la gestion des langues inconnues."""
        text = "Hello, world!"
        assert translate_text(text, src_lang="unknown", tgt_lang="en") == text

    def test_translate_text_auto_language_detection(self, mock_translation_service):
        """Vérifier la détection automatique de la langue source."""
        text = "Bonjour le monde !"
        translated = translate_text(text, tgt_lang="en")
        assert translated == "Translated text"

        # Vérifier que le service a été appelé avec la bonne langue source détectée
        mock_translation_service.translate.assert_called_once_with(
            text, source_lang="fr", target_lang="en"
        )