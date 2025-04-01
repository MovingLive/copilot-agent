"""Utilitaires pour la détection et traduction de langues.

Ce module fournit des fonctions pour détecter la langue d'un texte
et traduire du texte entre différentes langues en utilisant un modèle
multilingue léger.
"""

import logging

import langid

from app.core.config import settings
from app.services.translation_service import TranslationService

# Configuration du logger
logger = logging.getLogger(__name__)

# Configuration de langid pour une meilleure détection
langid.set_languages(
    ["en", "fr", "es", "de", "it", "pt", "nl", "ru", "ja", "zh", "ar", "ko", "hi"]
)


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

    # Mappage des codes de langue ISO pour NLLB
    nllb_lang_map = {
        "en": "eng_Latn",
        "fr": "fra_Latn",
        "es": "spa_Latn",
        "de": "deu_Latn",
        "it": "ita_Latn",
        "pt": "por_Latn",
        "nl": "nld_Latn",
        "ru": "rus_Cyrl",
        "ja": "jpn_Jpan",
        "zh": "zho_Hans",
        "ar": "arb_Arab",
        "ko": "kor_Hang",
        "hi": "hin_Deva",
    }

    # Conversion des codes de langue ISO vers les codes NLLB
    nllb_src_lang = nllb_lang_map.get(
        src_lang, "eng_Latn"
    )  # Défaut vers anglais si non trouvé
    nllb_tgt_lang = nllb_lang_map.get(tgt_lang, "eng_Latn")

    try:
        logger.debug("Traduction de '%s' à '%s'", src_lang, tgt_lang)

        # Utilisation du service de traduction préchargé au lieu de charger le modèle à la demande
        translation_service = TranslationService.get_instance()
        if not translation_service.is_loaded:
            logger.warning("Modèle de traduction non chargé, impossible de traduire")
            return text

        tokenizer, model = translation_service.model_and_tokenizer

        # Tokenization du texte avec le code de langue NLLB source
        encoded = tokenizer(text, return_tensors="pt")

        # Génération de la traduction avec le code de langue NLLB cible
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.lang_code_to_id[nllb_tgt_lang],
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
