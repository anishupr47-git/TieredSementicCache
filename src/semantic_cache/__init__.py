"""
Tiered Semantic Cache - Main Package & Public Python API
========================================================

What is this file?
------------------
This is the front door of our cache system.
You can import this library directly into any Python script, FastAPI web app,
Django backend, or AI chatbot!

Simple Math & Logic in Points:
------------------------------
1. Instant Exact Search (Fast Path):
   - If someone asks the exact same question word-for-word, the answer is
     returned instantly in ~1 microsecond.
   - It skips doing any math or vector conversions!

2. Smart Meaning Search (Semantic Path):
   - If the exact text isn't found, your question is converted into an "arrow" (vector).
   - The computer calculates the angle between your question's arrow and all saved answers.
   - If the meaning is close enough (e.g., >= 70% match), it returns the saved answer!

3. Multi-Tenant Protection (NamespacedSemanticCache):
   - Prevents User A (Ram) and User B (Shyam) from ever seeing each other's private data,
     even if they ask the exact same question!
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple

from semantic_cache.client import SemanticCacheClient
from semantic_cache.config import CacheConfig
from semantic_cache.embedder import BaseEmbedder, DenseHashEmbedder
from semantic_cache.storage.l1_ram import CacheRecord
from semantic_cache.storage.manager import LookupResult, StorageManager


class TieredSemanticCache:
    """High-performance two-tier (RAM + Disk) semantic cache."""

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        embedder: Optional[BaseEmbedder] = None,
    ) -> None:
        """Initialize the cache with settings and vector converter.

        Args:
            config: System settings (defaults to CacheConfig()).
            embedder: Tool that turns text into arrows (defaults to offline DenseHashEmbedder).
        """
        self.config = config or CacheConfig()
        self.embedder = embedder or DenseHashEmbedder(dim=self.config.vector_dim)
        self.storage = StorageManager(self.config)

    def get(self, query: str, _key_prefix: Optional[str] = None) -> Optional[LookupResult]:
        """Look up an answer for a question (exact or similar meaning).

        How it works:
        1. Checks RAM and Disk for an exact word-for-word match in 1 step.
           (Returns instantly without computing any vectors!).
        2. If missed, converts question to a vector arrow and compares meaning
           against all saved answers.

        Args:
            query: The user's question or prompt.
            _key_prefix: Internal. Restricts semantic search to keys starting with
                         this prefix (used by namespaced caches for tenant isolation).

        Returns:
            LookupResult with the answer and similarity score, or None if missed.
        """
        if not query or not isinstance(query, str):
            return None

        return self.storage.get(query, embed_fn=self.embedder.embed, key_prefix=_key_prefix)

    def set(
        self,
        query: str,
        answer: str,
        ttl: Optional[int] = None,
        tags: Sequence[str] = (),
    ) -> None:
        """Save a question and answer with an optional countdown timer (TTL) and tags.

        Puts the item in fast RAM. If RAM is full, automatically moves
        the oldest unused item into persistent disk storage.

        Args:
            query: The user's question.
            answer: The answer or LLM response to save.
            ttl: Optional countdown in seconds before expiration.
            tags: Optional labels (e.g. ['finance', 'user_settings']) for group cleanup.
        """
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if not isinstance(answer, str):
            raise ValueError("Answer must be a string.")

        vector = self.embedder.embed(query)
        self.storage.set(key=query, value=answer, vector=vector, ttl=ttl, tags=tags)

    def expire(self, query: str, ttl_seconds: float) -> bool:
        """Set or update a countdown timer on an existing cached answer."""
        if not query or not isinstance(query, str):
            return False
        return self.storage.expire(query, ttl_seconds)

    def ttl(self, query: str) -> int:
        """Check remaining seconds before an answer expires (-2 if missing, -1 if no timer)."""
        if not query or not isinstance(query, str):
            return -2
        return self.storage.ttl(query)

    def invalidate_tag(self, tag: str) -> int:
        """Delete all cached answers labeled with a specific tag."""
        if not tag or not isinstance(tag, str):
            return 0
        return self.storage.invalidate_tag(tag)

    def namespace(self, name: str) -> NamespacedSemanticCache:
        """Create a private room for a specific user or tenant."""
        if not name or not isinstance(name, str):
            raise ValueError("Namespace must be a non-empty string.")
        return NamespacedSemanticCache(self, name)

    def delete(self, query: str) -> bool:
        """Delete an answer completely from RAM and disk."""
        if not query or not isinstance(query, str):
            return False
        return self.storage.delete(query)

    def compact(self) -> int:
        """Clean up the disk file to reclaim wasted hard drive space."""
        return self.storage.compact()

    def sweep_expired(self) -> int:
        """Toss out all expired items across both RAM and Disk."""
        return self.storage.sweep_expired()

    def waste_stats(self) -> Tuple[int, int, float]:
        """PR-6: Check disk file waste stats: (total_size, wasted_bytes, waste_ratio)."""
        return self.storage.l2.waste_stats()

    def clear(self) -> None:
        """Wipe the entire cache clean."""
        self.storage.clear()

    def stats(self) -> Dict[str, Any]:
        """Check cache performance numbers (hits, misses, item counts)."""
        return self.storage.stats()

    def close(self) -> None:
        """Cleanly close storage files and background workers."""
        self.storage.close()

    def __enter__(self) -> TieredSemanticCache:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __len__(self) -> int:
        """Total number of saved answers across both RAM and Disk."""
        return len(self.storage)

    def __contains__(self, query: str) -> bool:
        """Check if a question exists in the cache."""
        if not isinstance(query, str):
            return False
        return query in self.storage

    def __getitem__(self, query: str) -> str:
        """Dictionary-style lookup: cache['question'] -> answer."""
        res = self.get(query)
        if res is None:
            raise KeyError(query)
        return res.value

    def __setitem__(self, query: str, answer: str) -> None:
        """Dictionary-style store: cache['question'] = answer."""
        self.set(query, answer)

    def __delitem__(self, query: str) -> None:
        """Dictionary-style delete: del cache['question']."""
        if not self.delete(query):
            raise KeyError(query)

    def __repr__(self) -> str:
        return (
            f"<TieredSemanticCache "
            f"l1={len(self.storage.l1)}/{self.config.ram_capacity}, "
            f"l2={len(self.storage.l2)}>"
        )


class NamespacedSemanticCache:
    """Multi-user isolated room: keeps each user's cached data completely separate."""

    def __init__(self, cache: TieredSemanticCache, namespace: str) -> None:
        self._cache = cache
        self.namespace = namespace.strip(":")

    def _prefix_key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, query: str) -> Optional[LookupResult]:
        """Look up an answer, strictly ensuring no other user's data can ever be seen."""
        prefix = f"{self.namespace}:"
        # SEC-4: Pre-filter vectors by namespace BEFORE the dot-product
        res = self._cache.get(self._prefix_key(query), _key_prefix=prefix)
        if res is None:
            return None

        # Security Guard: Double-check the match belongs to THIS user's namespace
        if not res.matched_key.startswith(prefix):
            return None

        clean_key = res.matched_key[len(prefix):]
        return LookupResult(
            value=res.value,
            similarity=res.similarity,
            matched_key=clean_key,
            tier=res.tier,
            ttl=res.ttl,
            tags=res.tags,
        )

    def set(
        self,
        query: str,
        answer: str,
        ttl: Optional[int] = None,
        tags: Sequence[str] = (),
    ) -> None:
        """Store an answer inside this user's private namespace."""
        ns_tags = [f"{self.namespace}:{t}" for t in tags]
        self._cache.set(self._prefix_key(query), answer, ttl=ttl, tags=ns_tags)

    def delete(self, query: str) -> bool:
        """Delete an answer from this user's private namespace."""
        return self._cache.delete(self._prefix_key(query))

    def expire(self, query: str, ttl_seconds: float) -> bool:
        """Set a timer on an answer inside this user's namespace."""
        return self._cache.expire(self._prefix_key(query), ttl_seconds)

    def ttl(self, query: str) -> int:
        """Check remaining seconds before an answer expires in this namespace."""
        return self._cache.ttl(self._prefix_key(query))

    def invalidate_tag(self, tag: str) -> int:
        """Delete all answers with this tag inside this user's namespace."""
        return self._cache.invalidate_tag(f"{self.namespace}:{tag}")

    def __contains__(self, query: str) -> bool:
        return self._prefix_key(query) in self._cache

    def __getitem__(self, query: str) -> str:
        res = self.get(query)
        if res is None:
            raise KeyError(query)
        return res.value

    def __setitem__(self, query: str, answer: str) -> None:
        self.set(query, answer)

    def __delitem__(self, query: str) -> None:
        if not self.delete(query):
            raise KeyError(query)

    def __repr__(self) -> str:
        return f"<NamespacedSemanticCache namespace='{self.namespace}'>"


__all__ = [
    "TieredSemanticCache",
    "NamespacedSemanticCache",
    "SemanticCacheClient",
    "CacheConfig",
    "LookupResult",
    "CacheRecord",
    "BaseEmbedder",
    "DenseHashEmbedder",
]
