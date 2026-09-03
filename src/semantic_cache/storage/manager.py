"""
Tiered Semantic Cache - Two-Tier Storage Manager (The "Office Manager")
========================================================================

What is this file?
------------------
This file coordinates our two tiers:
- Tier 1: L1 RAM (The Desk)
- Tier 2: L2 Disk (The Filing Cabinet)

Think of it as the smart Office Manager:
When someone asks for an answer:
1. First, check your Desk (L1 RAM). It's right in front of you (fastest!).
2. If not on your desk, check the Filing Cabinet (L2 Disk).
3. If found in the cabinet, promote it back to your desk so it is ready for next time!
4. If your desk gets full, slide the oldest paper into the cabinet so nothing is lost.

Lookup Flow in 4 Simple Steps:
------------------------------
1. L1 Exact Match (O(1)):
   - Did they ask the exact same question? Return answer immediately.
2. L1 Semantic Match (O(N*d)):
   - Does a question on your desk mean the same thing (score >= threshold)?
     Return it!
3. L2 Exact Match (O(1)):
   - Is it saved in the filing cabinet?
   - Read it with zero-copy mmap, promote it to L1, and return it.
4. L2 Semantic Match (O(M*d)):
   - Does a question in the cabinet mean the same thing?
   - Read it, promote it to L1, and return it.
5. Cache Miss:
   - Not found anywhere. Return None.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple
import numpy as np

from semantic_cache.config import CacheConfig
from semantic_cache.storage.l1_ram import CacheRecord, L1RAMCache
from semantic_cache.storage.l2_disk import L2DiskCache


@dataclass
class LookupResult:
    """The outcome of searching our two-tier cache."""

    value: str
    similarity: float
    matched_key: str
    tier: str  # "L1_EXACT", "L1_SEMANTIC", "L2_EXACT", "L2_SEMANTIC"


class StorageManager:
    """Orchestrates L1 in-memory cache and L2 persistent disk storage."""

    def __init__(self, config: CacheConfig) -> None:
        """Initialize both tiers using system configuration."""
        self.config = config
        self.l1 = L1RAMCache(capacity=config.ram_capacity, dim=config.vector_dim)
        self.l2 = L2DiskCache(file_path=config.disk_path, dim=config.vector_dim)

        # Performance counters
        self._l1_hits: int = 0
        self._l2_hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    def get(
        self,
        key: str,
        query_vector: Optional[np.ndarray] = None,
    ) -> Optional[LookupResult]:
        """Search the cache hierarchy: L1 Exact -> L1 Semantic -> L2 Exact -> L2 Semantic."""
        # 1. Check L1 Exact Hit in O(1)
        rec = self.l1.get_exact(key)
        if rec is not None:
            self._l1_hits += 1
            return LookupResult(
                value=rec.value,
                similarity=1.0,
                matched_key=rec.key,
                tier="L1_EXACT",
            )

        # 2. Check L1 Semantic Hit in O(N*d)
        if query_vector is not None:
            hit = self.l1.find_semantic(query_vector, self.config.similarity_threshold)
            if hit is not None:
                sem_rec, score = hit
                self._l1_hits += 1
                return LookupResult(
                    value=sem_rec.value,
                    similarity=score,
                    matched_key=sem_rec.key,
                    tier="L1_SEMANTIC",
                )

        # 3. Check L2 Exact Hit in O(1)
        rec = self.l2.get_exact(key)
        if rec is not None:
            self._l2_hits += 1
            # Promote cold item from disk back to RAM desk!
            self._promote_to_l1(rec)
            return LookupResult(
                value=rec.value,
                similarity=1.0,
                matched_key=rec.key,
                tier="L2_EXACT",
            )

        # 4. Check L2 Semantic Hit in O(M*d)
        if query_vector is not None:
            hit = self.l2.find_semantic(query_vector, self.config.similarity_threshold)
            if hit is not None:
                sem_rec, score = hit
                self._l2_hits += 1
                # Promote cold item from disk back to RAM desk!
                self._promote_to_l1(sem_rec)
                return LookupResult(
                    value=sem_rec.value,
                    similarity=score,
                    matched_key=sem_rec.key,
                    tier="L2_SEMANTIC",
                )

        # 5. Missed everywhere
        self._misses += 1
        return None

    def set(self, key: str, value: str, vector: np.ndarray) -> None:
        """Store an answer in L1 RAM. If L1 is full, spills oldest item to L2 Disk in O(1)."""
        evicted = self.l1.put(key, value, vector)

        # If an item was pushed off the desk, save it safely in the filing cabinet
        if evicted is not None:
            self._evictions += 1
            self.l2.append(evicted.key, evicted.value, evicted.vector)

    def _promote_to_l1(self, record: CacheRecord) -> None:
        """Move a cold item from L2 Disk back onto the warm L1 RAM desk."""
        evicted = self.l1.put(record.key, record.value, record.vector)
        if evicted is not None:
            self._evictions += 1
            self.l2.append(evicted.key, evicted.value, evicted.vector)

    def stats(self) -> dict[str, Any]:
        """Return system metrics: counts, hits, misses, and evictions."""
        return {
            "l1_count": len(self.l1),
            "l2_count": len(self.l2),
            "l1_capacity": self.config.ram_capacity,
            "l1_hits": self._l1_hits,
            "l2_hits": self._l2_hits,
            "total_hits": self._l1_hits + self._l2_hits,
            "misses": self._misses,
            "evictions": self._evictions,
        }

    def close(self) -> None:
        """Close storage handles cleanly."""
        self.l2.close()
