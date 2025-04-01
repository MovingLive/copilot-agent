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

        tokenizer_mock = MagicMock()
        tokenizer_mock.batch_decode.return_value = ["Translated text"]
        tokenizer_mock.lang_code_to_id = {
            "eng_Latn": 1,
            "fra_Latn": 2,
            "spa_Latn": 3,
            "deu_Latn": 4,
        }

        model_mock = MagicMock()
        model_mock.generate.return_value = "generated_tokens"

        service_mock.model_and_tokenizer = (tokenizer_mock, model_mock)
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
        translated = translate_text(
            "Bonjour le monde !", src_lang="fr", tgt_lang="en"
        )
        assert translated == "Translated text"

        # Vérifier que le service a été appelé avec les bons codes de langue NLLB
        tokenizer, model = mock_translation_service.model_and_tokenizer
        model.generate.assert_called_once()
        # Vérifier que l'ID de la langue cible a été utilisé
        called_args = model.generate.call_args[1]
        assert 'forced_bos_token_id' in called_args
        assert called_args['forced_bos_token_id'] == tokenizer.lang_code_to_id["eng_Latn"]

    def test_translate_text_service_not_loaded(self):
        """Vérifier le comportement lorsque le service de traduction n'est pas chargé."""
        with patch.object(TranslationService, "get_instance") as mock_get_instance:
            service_mock = MagicMock()
            service_mock.is_loaded = False
            mock_get_instance.return_value = service_mock

            original = "Bonjour le monde !"
            assert translate_text(original, src_lang="fr", tgt_lang="en") == original

    def test_translate_text_error_handling(self):
        """Vérifier la gestion des erreurs lors de la traduction."""
        with patch.object(TranslationService, "get_instance") as mock_get_instance:
            service_mock = MagicMock()
            service_mock.is_loaded = True
            service_mock.model_and_tokenizer = (MagicMock(), MagicMock())
            service_mock.model_and_tokenizer[1].generate.side_effect = Exception("Test error")
            mock_get_instance.return_value = service_mock

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

    def test_translate_text_nllb_language_mapping(self, mock_translation_service):
        """Vérifier le mappage correct des codes de langue ISO vers NLLB."""
        lang_pairs = [
            ("fr", "en", "fra_Latn", "eng_Latn"),
            ("es", "de", "spa_Latn", "deu_Latn"),
        ]

        for src_iso, tgt_iso, src_nllb, tgt_nllb in lang_pairs:
            translate_text("Test text", src_lang=src_iso, tgt_lang=tgt_iso)

            # Vérifier que le bon code NLLB a été utilisé
            tokenizer, _ = mock_translation_service.model_and_tokenizer
            called_args = mock_translation_service.model_and_tokenizer[1].generate.call_args[1]
            assert called_args['forced_bos_token_id'] == tokenizer.lang_code_to_id[tgt_nllb]