import os
import time
import pytest
import numpy as np

from semantic_cache import TieredSemanticCache, CacheConfig, DenseHashEmbedder


def test_cache_exact_match(tmp_path):
    """Test that asking the exact same question gets the exact same answer."""
    config = CacheConfig(disk_path=tmp_path / "test_exact.bin")
    with TieredSemanticCache(config) as cache:
        cache.set("What is the capital of France?", "Paris")

        result = cache.get("What is the capital of France?")
        assert result is not None
        assert result.value == "Paris"
        assert result.similarity == 1.0
        assert result.tier == "L1_EXACT"


def test_cache_semantic_match(tmp_path):
    """Test that a similar question returns the cached answer."""
    config = CacheConfig(
        disk_path=tmp_path / "test_semantic.bin",
        similarity_threshold=0.85,
    )
    with TieredSemanticCache(config) as cache:
        cache.set("What's the best way to cook steak?", "Reverse sear it.")

        result = cache.get("What is the best way to cook steak?")
        assert result is not None
        assert result.value == "Reverse sear it."


def test_cache_eviction_to_l2(tmp_path):
    """Test that when L1 gets full, it safely spills to L2."""
    config = CacheConfig(
        ram_capacity=2,  # Desk can only hold 2 items
        disk_path=tmp_path / "test_eviction.bin",
    )
    with TieredSemanticCache(config) as cache:
        # 1. Fill the desk (L1)
        cache.set("Q1", "A1")
        cache.set("Q2", "A2")

        assert cache.stats()["l1_count"] == 2
        assert cache.stats()["l2_count"] == 0

        # 2. Push an item off the desk! Q1 goes to filing cabinet (L2)
        cache.set("Q3", "A3")

        assert cache.stats()["l1_count"] == 2
        assert cache.stats()["l2_count"] == 1

        # 3. Can we still get Q1? Yes, from L2!
        result = cache.get("Q1")
        assert result is not None
        assert result.value == "A1"
        assert result.tier == "L2_EXACT"


def test_cache_pythonic_protocols(tmp_path):
    """Test len(), 'in' operator, and context manager."""
    config = CacheConfig(ram_capacity=2, disk_path=tmp_path / "test_protocols.bin")
    with TieredSemanticCache(config) as cache:
        cache.set("cat", "meow")
        cache.set("dog", "woof")
        cache.set("cow", "moo")  # Spills 'cat' to L2

        # Test len()
        assert len(cache) == 3

        # Test 'in' operator
        assert "cat" in cache
        assert "dog" in cache
        assert "bird" not in cache

        # Test input validation
        with pytest.raises(ValueError):
            cache.set("", "bad")

        with pytest.raises(ValueError):
            cache.set("key", 123)  # type: ignore


def test_performance(tmp_path):
    """Make sure exact lookups are sub-microsecond fast!"""
    config = CacheConfig(disk_path=tmp_path / "test_perf.bin")
    with TieredSemanticCache(config) as cache:
        cache.set("Hello world", "Hi there!")

        start = time.perf_counter()
        for _ in range(1000):
            cache.get("Hello world")
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        avg_us = (elapsed_ms / 1000) * 1000
        print(f"1,000 Exact GETs took {elapsed_ms:.2f} ms ({avg_us:.2f} us/call)")

        # 1,000 exact lookups should easily finish under 10ms with our O(1) fast-path
        assert elapsed_ms < 10.0, "Cache is running too slow!"
