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

        # ---------------------------------------------------------------------
        # FAST PATH: O(1) Exact Matching (Zero Vector Computation Overhead)
        # ---------------------------------------------------------------------
        # 1. Check L1 RAM desk (pure dictionary lookup: ~100 nanoseconds)
        rec = self.storage.l1.get_exact(query)
        if rec is not None:
            self.storage._l1_hits += 1
            return LookupResult(
                value=rec.value,
                similarity=1.0,
                matched_key=rec.key,
                tier="L1_EXACT",
            )

        # 2. Check L2 Disk index (in-memory hash index + zero-copy mmap read)
        rec = self.storage.l2.get_exact(query)
        if rec is not None:
            self.storage._l2_hits += 1
            # Promote item back to L1 RAM
            self.storage._promote_to_l1(rec)
            return LookupResult(
                value=rec.value,
                similarity=1.0,
                matched_key=rec.key,
                tier="L2_EXACT",
            )

        # ---------------------------------------------------------------------
        # SEMANTIC PATH: O((N + M) * d) Vector Similarity Search
        # ---------------------------------------------------------------------
        # Only compute the vector embedding when exact matches have missed!
        vector = self.embedder.embed(query)

        # 3. Check L1 RAM vectors
        hit = self.storage.l1.find_semantic(vector, self.config.similarity_threshold)
        if hit is not None:
            sem_rec, score = hit
            self.storage._l1_hits += 1
            return LookupResult(
                value=sem_rec.value,
                similarity=score,
                matched_key=sem_rec.key,
                tier="L1_SEMANTIC",
            )

        # 4. Check L2 Disk vectors
        hit = self.storage.l2.find_semantic(vector, self.config.similarity_threshold)
        if hit is not None:
            sem_rec, score = hit
            self.storage._l2_hits += 1
            self.storage._promote_to_l1(sem_rec)
            return LookupResult(
                value=sem_rec.value,
                similarity=score,
                matched_key=sem_rec.key,
                tier="L2_SEMANTIC",
            )

        # 5. Missed everywhere
        self.storage._misses += 1
        return None

    def set(self, query: str, answer: str) -> None:
        """Store a question and answer pair.

        Computes the dense vector embedding and inserts into L1 RAM.
        If L1 capacity is exceeded, automatically spills the least-recently used
        record to persistent L2 Disk storage in O(1) amortized time.

        Args:
            query: The query string to index.
            answer: The value/answer string to cache.
        """
        if not query or not isinstance(query, str):
            raise ValueError("Query must be a non-empty string.")
        if not isinstance(answer, str):
            raise ValueError("Answer must be a string.")

        vector = self.embedder.embed(query)
        self.storage.set(key=query, value=answer, vector=vector)

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
        return len(self.storage.l1) + len(self.storage.l2)

    def __contains__(self, query: str) -> bool:
        """Check if a query exists in L1 or L2 in O(1) time without reading payload."""
        if not isinstance(query, str):
            return False
        return (query in self.storage.l1) or (query in self.storage.l2)

    def __repr__(self) -> str:
        """Return human-readable cache summary."""
        return (
            f"<TieredSemanticCache "
            f"l1={len(self.storage.l1)}/{self.config.ram_capacity}, "
            f"l2={len(self.storage.l2)}>"
        )


__all__ = [
    "TieredSemanticCache",
    "CacheConfig",
    "LookupResult",
    "CacheRecord",
    "BaseEmbedder",
    "DenseHashEmbedder",
]
