"""Utilitaires pour la détection et traduction de langues.

Ce module fournit des fonctions pour détecter la langue d'un texte
et traduire du texte entre différentes langues en utilisant un modèle
multilingue (M2M100).
"""

import logging
from functools import lru_cache

import langid
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

from app.core.config import settings

# Configuration du logger
logger = logging.getLogger(__name__)

# Configuration de langid pour une meilleure détection
langid.set_languages(
    ["en", "fr", "es", "de", "it", "pt", "nl", "ru", "ja", "zh", "ar", "ko", "hi"]
)


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
            "need",
            "help",
            "can",
            "get",
            "has",
            "about",
            "how",
            "why",
            "when",
            "what",
            "where",
            "who",
            "which",
            "me",
            "us",
            "am",
        }

        # Liste de mots français courants pour validation supplémentaire
        french_common_words = {
            "le",
            "la",
            "les",
            "un",
            "une",
            "des",
            "et",
            "est",
            "sont",
            "ce",
            "cette",
            "ces",
            "je",
            "tu",
            "il",
            "elle",
            "nous",
            "vous",
            "ils",
            "elles",
            "mon",
            "ton",
            "son",
            "notre",
            "votre",
            "leur",
            "pour",
            "avec",
            "sans",
            "dans",
            "sur",
            "sous",
            "entre",
            "qui",
            "que",
            "quoi",
            "comment",
            "pourquoi",
            "quand",
            "où",
            "ceci",
            "cela",
            "ai",
            "as",
            "avons",
            "avez",
            "ont",
            "suis",
            "es",
            "sommes",
            "êtes",
        }

        # Nettoyage et préparation du texte
        words = [word.lower() for word in text.split()]
        word_set = set(words)

        # Gestion spéciale pour les phrases très courtes (moins de 4 mots)
        if len(words) < 4:
            # Vérifier si des mots français connus sont présents
            french_word_count = len(word_set.intersection(french_common_words))
            if french_word_count >= 1:
                logger.debug(
                    "Phrase courte détectée comme français: %s mots français courants",
                    french_word_count,
                )
                return "fr"

            # Vérifier si des mots anglais connus sont présents
            english_word_count = len(word_set.intersection(english_common_words))
            if english_word_count >= 1:
                logger.debug(
                    "Phrase courte détectée comme anglais: %s mots anglais courants",
                    english_word_count,
                )
                return "en"

            # Vérification pour les phrases typiquement anglaises
            common_english_phrases = {"i need", "i am", "i have", "help me", "can you"}
            lower_text = text.lower()
            if any(phrase in lower_text for phrase in common_english_phrases):
                logger.debug("Phrase anglaise courante détectée")
                return "en"

            # Vérification pour les phrases typiquement françaises
            common_french_phrases = {
                "je suis",
                "j'ai",
                "aidez-moi",
                "pouvez-vous",
                "ceci est",
            }
            if any(phrase in lower_text for phrase in common_french_phrases):
                logger.debug("Phrase française courante détectée")
                return "fr"
        else:
            # Pour les phrases plus longues, vérifier d'abord les mots connus
            french_word_count = len(word_set.intersection(french_common_words))
            if french_word_count >= 2:
                logger.debug(
                    "Texte détecté comme français: %s mots français courants",
                    french_word_count,
                )
                return "fr"

            english_word_count = len(word_set.intersection(english_common_words))
            if english_word_count >= 2:
                logger.debug(
                    "Texte détecté comme anglais: %s mots anglais courants",
                    english_word_count,
                )
                return "en"

        # Utiliser langid pour la détection de langue
        lang, confidence = langid.classify(text)
        logger.debug("Langue détectée par langid: %s (confiance: %s)", lang, confidence)

        # Si la confiance est faible, appliquer des règles additionnelles
        if confidence < 0.5:
            # Vérifier si d'autres indicateurs peuvent nous aider
            if any(word in french_common_words for word in word_set):
                logger.debug(
                    "Correction de la détection à faible confiance vers français"
                )
                return "fr"
            if any(word in english_common_words for word in word_set):
                logger.debug(
                    "Correction de la détection à faible confiance vers anglais"
                )
                return "en"

        return lang
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
