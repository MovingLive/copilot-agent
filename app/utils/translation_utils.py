"""Utilitaires pour la détection et traduction de langues.

Ce module fournit des fonctions pour détecter la langue d'un texte
et traduire du texte entre différentes langues en utilisant un modèle
multilingue (M2M100).
"""

import logging
from functools import lru_cache

import langdetect
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from app.core.config import settings

# Configuration du logger
logger = logging.getLogger(__name__)

# Configuration de langdetect pour une détection stable
langdetect.DetectorFactory.seed = 0


@lru_cache(maxsize=1)
def _get_translation_model_and_tokenizer():
    """Charge et met en cache le modèle de traduction M2M100 et son tokenizer.

    Returns:
        tuple: (tokenizer, model) pour M2M100
    """
    try:
        # Utilisation d'une version plus légère du modèle M2M100
        model_name = "facebook/m2m100_418M"  # Version plus légère que le modèle complet
        logger.info("Chargement du modèle de traduction %s", model_name)

        tokenizer = M2M100Tokenizer.from_pretrained(model_name)
        model = M2M100ForConditionalGeneration.from_pretrained(model_name)

        logger.info("Modèle de traduction chargé avec succès")
        return tokenizer, model
    except Exception as e:
        logger.error("Erreur lors du chargement du modèle de traduction: %s", e)
        raise


def detect_language(text: str) -> str:
    """Détecte la langue d'un texte.

    Args:
        text: Le texte dont on veut détecter la langue.

    Returns:
        str: Code de langue ISO (ex: 'fr', 'en', etc.)

    Raises:
        ValueError: Si la détection de langue échoue ou si le texte est vide.
    """
    if not text or not text.strip():
        raise ValueError("Le texte ne peut pas être vide")

    try:
        # Liste de mots anglais courants pour validation supplémentaire
        english_common_words = {
            "the",
            "be",
            "to",
            "of",
            "and",
            "a",
            "in",
            "that",
            "have",
            "i",
            "it",
            "for",
            "not",
            "on",
            "with",
            "he",
            "as",
            "you",
            "do",
            "at",
            "this",
            "but",
            "his",
            "by",
            "from",
            "they",
            "we",
            "say",
            "her",
            "she",
            "or",
            "an",
            "will",
            "my",
            "one",
            "all",
            "would",
            "there",
            "their",
        }

        # Nettoyage et préparation du texte
        words = set(word.lower() for word in text.split())

        # Si le texte contient plusieurs mots anglais communs, c'est probablement de l'anglais
        if len(words.intersection(english_common_words)) >= 2:
            return "en"

        # Utilisation de langdetect comme fallback
        detected = langdetect.detect(text)

        # Validation supplémentaire pour l'anglais
        if detected == "so" and any(word in english_common_words for word in words):
            return "en"

        return detected
    except Exception as e:
        logger.error("Erreur de détection de langue: %s", e)
        return "unknown"


def needs_translation(text: str, target_lang: str = None) -> bool:
    """Détermine si un texte doit être traduit.

    Args:
        text: Le texte à analyser
        target_lang: La langue cible (par défaut: FAISS_LANG des paramètres)

    Returns:
        bool: True si une traduction est nécessaire, False sinon
    """
    if not text or not text.strip():
        return False

    target_lang = target_lang or settings.FAISS_LANG
    detected_lang = detect_language(text)

    # Si la langue détectée est 'unknown', on ne traduit pas
    if detected_lang == "unknown":
        return False

    return detected_lang != target_lang


def translate_text(text: str, src_lang: str = None, tgt_lang: str = None) -> str:
    """Traduit un texte de la langue source vers la langue cible.

    Args:
        text: Le texte à traduire
        src_lang: La langue source (si None, sera détectée automatiquement)
        tgt_lang: La langue cible (par défaut: FAISS_LANG des paramètres)

    Returns:
        str: Le texte traduit

    Raises:
        ValueError: Si la traduction échoue
    """
    if not text or not text.strip():
        return text

    # Détection de langue si non spécifiée
    if not src_lang:
        src_lang = detect_language(text)

    # Définition de la langue cible par défaut
    if not tgt_lang:
        tgt_lang = settings.FAISS_LANG

    # Si mêmes langues ou langue inconnue, retourner le texte original
    if src_lang == "unknown" or src_lang == tgt_lang:
        return text

    try:
        logger.debug("Traduction de '%s' à '%s'", src_lang, tgt_lang)
        tokenizer, model = _get_translation_model_and_tokenizer()

        # Configuration du tokenizer pour la langue source
        tokenizer.src_lang = src_lang

        # Tokenization du texte
        encoded = tokenizer(text, return_tensors="pt")

        # Génération de la traduction
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(tgt_lang),
            max_length=1024,
        )

        # Décodage de la traduction
        translated_text = tokenizer.batch_decode(
            generated_tokens, skip_special_tokens=True
        )[0]

        return translated_text
    except Exception as e:
        logger.error("Erreur lors de la traduction: %s", e)
        # En cas d'erreur, retourner le texte original
        return text
