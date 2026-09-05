"""
Tiered Semantic Cache - Public SDK Interface
============================================

The simple, high-performance in-process SDK for Python applications.
Ideal for shared hosting (e.g., cPanel, AWS Lambda, WSGI/ASGI apps)
where running a background TCP daemon is not possible.

Time & Space Complexity Guarantees:
-----------------------------------
- get(query):
    * Fast Path (Exact Hit):    O(1) time, 0 allocations (skips vector embedding entirely).
    * Semantic Path (Fuzzy Hit): O(L + (N + M) * d) where:
        - L = length of query string (for subword hashing)
        - N = items in L1 RAM
        - M = items in L2 Disk
        - d = vector dimension (e.g. 384)
- set(query, answer):
    * O(L + d) to embed and insert into L1 (O(1) amortized disk spill if L1 is full).
- __contains__(query):
    * O(1) hash check across L1 and L2 memory indices.
- __len__():
    * O(1) count of total cached records.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from semantic_cache.client import SemanticCacheClient
from semantic_cache.config import CacheConfig
from semantic_cache.embedder import BaseEmbedder, DenseHashEmbedder
from semantic_cache.storage.l1_ram import CacheRecord
from semantic_cache.storage.manager import LookupResult, StorageManager


class TieredSemanticCache:
    """High-performance in-process Semantic Cache.

    Coordinates vector embedding with two-tier (RAM + Disk) storage.
    Includes fast-path exact matching to bypass vector calculation when possible,
    reducing read latency to sub-microsecond levels.
    """

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        embedder: Optional[BaseEmbedder] = None,
    ) -> None:
        """Initialize the cache with configuration and vector embedder.

        Args:
            config: Cache configuration (defaults to CacheConfig()).
            embedder: Pluggable vector embedder (defaults to DenseHashEmbedder).
        """
        self.config = config or CacheConfig()
        self.embedder = embedder or DenseHashEmbedder(dim=self.config.vector_dim)
        self.storage = StorageManager(self.config)

    def get(self, query: str) -> Optional[LookupResult]:
        """Retrieve an answer for a query using two-tier exact and semantic lookup.

        Latency Optimization:
        1. Fast Path O(1): Checks L1 RAM dictionary for exact match.
        2. Fast Path O(1): Checks L2 Disk index for exact match.
           -> If exact match found, returns immediately WITHOUT computing embedding!
        3. Semantic Path O((N+M)*d): Computes vector embedding only on exact miss
           and scans L1 and L2 vector matrices for cosine similarity.

        Args:
            query: The user query or question.

        Returns:
            LookupResult if found (with value, similarity score, matched key, and tier),
            or None if missed.
        """
        if not query or not isinstance(query, str):
            return None

        return self.storage.get(query, embed_fn=self.embedder.embed)

    def set(
        self,
        query: str,
        answer: str,
        ttl: Optional[int] = None,
        tags: Sequence[str] = (),
    ) -> None:
        """Store a question and answer pair with optional TTL and tags.

        Computes the dense vector embedding and inserts into L1 RAM.
        If L1 capacity is exceeded, automatically spills the least-recently used
        record to persistent L2 Disk storage in O(1) amortized time.

        Args:
            query: The query string to index.
            answer: The value/answer string to cache.
            ttl: Optional time-to-live in seconds (None uses config default or immortal).
            tags: Optional categorization tags for group invalidation.
        """
        if not query or not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string.")
        if not isinstance(answer, str):
            raise ValueError("Answer must be a string.")

        vector = self.embedder.embed(query)
        self.storage.set(key=query, value=answer, vector=vector, ttl=ttl, tags=tags)

    def expire(self, query: str, ttl_seconds: float) -> bool:
        """Set or update TTL expiration in seconds on an existing cached query."""
        if not query or not isinstance(query, str):
            return False
        return self.storage.expire(query, ttl_seconds)

    def ttl(self, query: str) -> int:
        """Return remaining TTL in seconds (-2 if missing, -1 if no expiry, >=0 remaining)."""
        if not query or not isinstance(query, str):
            return -2
        return self.storage.ttl(query)

    def invalidate_tag(self, tag: str) -> int:
        """Invalidate and purge all cached answers associated with a tag.

        Returns:
            Number of records purged.
        """
        if not tag or not isinstance(tag, str):
            return 0
        return self.storage.invalidate_tag(tag)

    def namespace(self, name: str) -> NamespacedSemanticCache:
        """Create an isolated, multi-tenant sub-view of the cache."""
        if not name or not isinstance(name, str):
            raise ValueError("Namespace must be a non-empty string.")
        return NamespacedSemanticCache(self, name)

    def delete(self, query: str) -> bool:
        """Delete an item from L1 and/or L2 in strict O(1) time.

        Args:
            query: The query string to remove.

        Returns:
            True if the item was found and removed, False otherwise.
        """
        if not query or not isinstance(query, str):
            return False
        return self.storage.delete(query)

    def compact(self) -> int:
        """Compact the L2 append-only disk log by purging dead/overwritten records.

        Returns:
            Number of bytes reclaimed from disk.
        """
        return self.storage.compact()

    def clear(self) -> None:
        """Clear all cached records in both L1 RAM and L2 Disk."""
        self.storage.clear()

    def stats(self) -> Dict[str, Any]:
        """Return operational cache metrics (counts, hits, misses, evictions)."""
        return self.storage.stats()

    def close(self) -> None:
        """Safely release underlying storage and memory-mapped file handles."""
        self.storage.close()

    def __enter__(self) -> TieredSemanticCache:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit with guaranteed resource cleanup."""
        self.close()

    def __len__(self) -> int:
        """Return total number of unique items across L1 and L2 tiers."""
        return len(self.storage)

    def __contains__(self, query: str) -> bool:
        """Check if a query exists in L1 or L2 in O(1) time without reading payload."""
        if not isinstance(query, str):
            return False
        return query in self.storage

    def __getitem__(self, query: str) -> str:
        """Retrieve answer string or raise KeyError if missed."""
        res = self.get(query)
        if res is None:
            raise KeyError(query)
        return res.value

    def __setitem__(self, query: str, answer: str) -> None:
        """Store query and answer pair via subscription syntax."""
        self.set(query, answer)

    def __delitem__(self, query: str) -> None:
        """Delete query via subscription syntax or raise KeyError if missing."""
        if not self.delete(query):
            raise KeyError(query)

    def __repr__(self) -> str:
        """Return human-readable cache summary."""
        return (
            f"<TieredSemanticCache "
            f"l1={len(self.storage.l1)}/{self.config.ram_capacity}, "
            f"l2={len(self.storage.l2)}>"
        )


class NamespacedSemanticCache:
    """Isolated multi-tenant view of the cache scoped by a namespace prefix."""

    def __init__(self, cache: TieredSemanticCache, namespace: str) -> None:
        self._cache = cache
        self.namespace = namespace.strip(":")

    def _prefix_key(self, key: str) -> str:
        return f"{self.namespace}:{key}"

    def get(self, query: str) -> Optional[LookupResult]:
        """Retrieve answer within this namespace scope, ensuring strict cross-tenant isolation."""
        res = self._cache.get(self._prefix_key(query))
        if res is None:
            return None

        prefix = f"{self.namespace}:"
        # Ensure semantic match belongs to this namespace
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
        """Store an answer within this namespace scope."""
        ns_tags = [f"{self.namespace}:{t}" for t in tags]
        self._cache.set(self._prefix_key(query), answer, ttl=ttl, tags=ns_tags)

    def delete(self, query: str) -> bool:
        """Delete query from this namespace scope."""
        return self._cache.delete(self._prefix_key(query))

    def expire(self, query: str, ttl_seconds: float) -> bool:
        """Update TTL for a query within this namespace."""
        return self._cache.expire(self._prefix_key(query), ttl_seconds)

    def ttl(self, query: str) -> int:
        """Return remaining TTL for a query within this namespace."""
        return self._cache.ttl(self._prefix_key(query))

    def invalidate_tag(self, tag: str) -> int:
        """Invalidate all items with tag within this namespace."""
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
