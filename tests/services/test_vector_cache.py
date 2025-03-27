"""Tests pour le service VectorCache.

Ce module contient les tests unitaires pour le service de cache vectoriel.
"""

import time
from unittest.mock import patch
import numpy as np
import pytest

from app.services.vector_cache import VectorCache, get_cache_instance


@pytest.fixture
def vector_cache():
    """Fixture qui fournit une instance propre de VectorCache pour chaque test."""
    cache = VectorCache(max_size=2, ttl_seconds=1)
    yield cache
    cache.clear()


def test_embedding_cache_basic(vector_cache):
    """Test les opérations basiques du cache d'embeddings."""
    text = "test text"
    vector = np.array([1.0, 2.0, 3.0])
    
    # Test cache miss initial
    assert vector_cache.get_embedding(text) is None
    
    # Test stockage et récupération
    vector_cache.store_embedding(text, vector)
    cached_vector = vector_cache.get_embedding(text)
    assert np.array_equal(cached_vector, vector)


def test_results_cache_basic(vector_cache):
    """Test les opérations basiques du cache de résultats."""
    query = "test query"
    results = [{"id": 1, "score": 0.9}]
    
    # Test cache miss initial
    assert vector_cache.get_search_results(query) is None
    
    # Test stockage et récupération
    vector_cache.store_search_results(query, results)
    cached_results = vector_cache.get_search_results(query)
    assert cached_results == results


def test_cache_eviction(vector_cache):
    """Test l'éviction LRU du cache quand il atteint sa taille maximale."""
    # Remplir le cache au-delà de sa capacité (max_size=2)
    texts = ["text1", "text2", "text3"]
    vectors = [np.array([i]) for i in range(3)]
    
    for text, vector in zip(texts, vectors):
        vector_cache.store_embedding(text, vector)
    
    # Le premier élément devrait avoir été évincé
    assert vector_cache.get_embedding("text1") is None
    assert vector_cache.get_embedding("text2") is not None
    assert vector_cache.get_embedding("text3") is not None


def test_ttl_expiration(vector_cache):
    """Test l'expiration TTL des entrées du cache."""
    query = "test query"
    results = [{"id": 1, "score": 0.9}]
    
    vector_cache.store_search_results(query, results)
    assert vector_cache.get_search_results(query) == results
    
    # Attendre que le TTL expire (ttl_seconds=1)
    time.sleep(1.1)
    assert vector_cache.get_search_results(query) is None


def test_cache_stats(vector_cache):
    """Test la collecte des statistiques du cache."""
    text = "test text"
    vector = np.array([1.0, 2.0, 3.0])
    
    # Générer quelques hits et misses
    vector_cache.get_embedding(text)  # Miss
    vector_cache.store_embedding(text, vector)
    vector_cache.get_embedding(text)  # Hit
    
    stats = vector_cache.get_cache_stats()
    assert stats["hit_count"] == 1
    assert stats["miss_count"] == 1
    assert "hit_rate" in stats
    assert "embedding_cache_size" in stats
    assert "results_cache_size" in stats


def test_cache_clear(vector_cache):
    """Test le nettoyage complet du cache."""
    text = "test text"
    vector = np.array([1.0, 2.0, 3.0])
    
    vector_cache.store_embedding(text, vector)
    vector_cache.clear()
    
    assert vector_cache.get_embedding(text) is None
    stats = vector_cache.get_cache_stats()
    assert stats["embedding_cache_size"] == 0
    assert stats["results_cache_size"] == 0


def test_singleton_pattern():
    """Test que get_cache_instance retourne toujours la même instance."""
    cache1 = get_cache_instance()
    cache2 = get_cache_instance()
    assert cache1 is cache2


@pytest.mark.parametrize("text,expected_hash", [
    ("test", "098f6bcd4621d373cade4e832627b4f6"),
    ("", "d41d8cd98f00b204e9800998ecf8427e"),
    ("12345", "827ccb0eea8a706c4c34a16891f84e7b"),
])
def test_text_hashing(vector_cache, text, expected_hash):
    """Test la génération cohérente des hashs de texte."""
    generated_hash = vector_cache._hash_text(text)
    assert generated_hash == expected_hash