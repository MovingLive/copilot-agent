"""Tests pour le service de traduction."""

import pytest
from unittest.mock import patch, MagicMock
from app.services.translation_service import TranslationService


@pytest.fixture(autouse=True)
def reset_singleton():
    """Fixture pour réinitialiser le singleton avant chaque test."""
    TranslationService._instance = None
    TranslationService._translator = None
    yield
    TranslationService._instance = None
    TranslationService._translator = None


@pytest.fixture
def translation_service():
    """Fixture pour le service de traduction."""
    # Réinitialiser le singleton entre chaque test
    TranslationService._instance = None
    TranslationService._translator = None
    return TranslationService.get_instance()


def test_singleton_pattern():
    """Test du pattern singleton."""
    service1 = TranslationService.get_instance()
    service2 = TranslationService.get_instance()
    assert service1 is service2


def test_is_loaded_false_by_default():
    """Test que le traducteur n'est pas chargé par défaut."""
    service = TranslationService.get_instance()
    assert not service.is_loaded


@patch('app.services.translation_service.GoogleTranslator')
def test_load_model(mock_translator):
    """Test du chargement du modèle."""
    service = TranslationService.get_instance()
    service.load_model()

    mock_translator.assert_called_once_with(source='auto', target='en')
    assert service.is_loaded


@patch('app.services.translation_service.GoogleTranslator')
def test_load_model_with_custom_langs(mock_translator):
    """Test du chargement du modèle avec des langues personnalisées."""
    service = TranslationService.get_instance()
    service.load_model(source_lang='fr', target_lang='es')

    mock_translator.assert_called_once_with(source='fr', target='es')
    assert service.is_loaded


@patch('app.services.translation_service.GoogleTranslator')
def test_load_model_error_handling(mock_translator):
    """Test de la gestion des erreurs lors du chargement du modèle."""
    mock_translator.side_effect = Exception("Test error")
    service = TranslationService.get_instance()

    with pytest.raises(Exception):
        service.load_model()

    assert not service.is_loaded


@patch('app.services.translation_service.GoogleTranslator')
def test_translate_loads_model_if_needed(mock_translator):
    """Test que la traduction charge le modèle si nécessaire."""
    mock_instance = MagicMock()
    mock_instance.translate.return_value = "Hello world"
    mock_translator.return_value = mock_instance

    service = TranslationService.get_instance()
    result = service.translate("Bonjour monde")

    assert result == "Hello world"
    mock_translator.assert_called_once()
    mock_instance.translate.assert_called_once_with("Bonjour monde")


@patch('app.services.translation_service.GoogleTranslator')
def test_translate_with_different_languages(mock_translator):
    """Test de la traduction avec différentes langues."""
    mock_instance = MagicMock()
    mock_instance.translate.return_value = "Hola mundo"
    mock_translator.return_value = mock_instance

    service = TranslationService.get_instance()
    result = service.translate("Hello world", source_lang='en', target_lang='es')

    assert result == "Hola mundo"
    mock_translator.assert_called_with(source='en', target='es')


@patch('app.services.translation_service.GoogleTranslator')
def test_translate_error_handling(mock_translator):
    """Test de la gestion des erreurs lors de la traduction."""
    mock_instance = MagicMock()
    mock_instance.translate.side_effect = Exception("Translation error")
    mock_translator.return_value = mock_instance

    service = TranslationService.get_instance()
    original_text = "Test text"
    result = service.translate(original_text)

    assert result == original_text  # Retourne le texte original en cas d'erreur


@patch('app.services.translation_service.GoogleTranslator')
def test_translate_updates_translator_if_languages_change(mock_translator):
    """Test que le traducteur est mis à jour si les langues changent."""
    mock_instance1 = MagicMock(source='auto', target='en')
    mock_instance2 = MagicMock(source='fr', target='es')
    mock_translator.side_effect = [mock_instance1, mock_instance2]

    service = TranslationService.get_instance()
    service.translate("Test")  # Premier appel avec langues par défaut
    service.translate("Test", source_lang='fr', target_lang='es')  # Second appel avec nouvelles langues

    assert mock_translator.call_count == 2
    mock_translator.assert_any_call(source='auto', target='en')
    mock_translator.assert_any_call(source='fr', target='es')