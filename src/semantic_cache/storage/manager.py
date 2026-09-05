from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import threading
import time
from typing import Any, Callable, Optional, Sequence, Tuple
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
    ttl: int = -1
    tags: Tuple[str, ...] = ()


class StorageManager:
    """Orchestrates L1 in-memory cache and L2 persistent disk storage with TTL & Tagging."""

    def __init__(self, config: CacheConfig) -> None:
        """Initialize both tiers using system configuration."""
        self.config = config
        self.l1 = L1RAMCache(capacity=config.ram_capacity, dim=config.vector_dim)
        self.l2 = L2DiskCache(file_path=config.disk_path, dim=config.vector_dim)
        self._lock = threading.RLock()

        # Tag indexing structures
        self._tag_to_keys: dict[str, set[str]] = defaultdict(set)
        self._key_to_tags: dict[str, set[str]] = defaultdict(set)

        # Performance counters
        self._l1_hits: int = 0
        self._l2_hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._expired_purges: int = 0

        # Background active expiration sweeper
        self._stop_sweep = threading.Event()
        self._sweep_thread: Optional[threading.Thread] = None
        if self.config.enable_active_sweep:
            self._sweep_thread = threading.Thread(
                target=self._active_sweep_loop,
                name="tsc-expiry-sweeper",
                daemon=True,
            )
            self._sweep_thread.start()

    def _active_sweep_loop(self) -> None:
        """Background daemon thread periodically sweeping expired records."""
        while not self._stop_sweep.wait(timeout=self.config.sweep_interval_sec):
            try:
                self.sweep_expired()
            except Exception:
                pass

    def get(
        self,
        key: str,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        query_vector: Optional[np.ndarray] = None,
    ) -> Optional[LookupResult]:
        """Search the cache hierarchy: L1 Exact -> L2 Exact -> L1 Semantic -> L2 Semantic.

        Fast-Path Optimization:
        Checks L1 and L2 for exact string matches first in O(1) time.
        Only computes vector embedding when both exact checks have missed!
        """
        with self._lock:
            # -----------------------------------------------------------------
            # FAST PATH: O(1) Exact Lookups (Zero Vector Calculations)
            # -----------------------------------------------------------------
            # 1. Check L1 Exact Hit in O(1)
            rec = self.l1.get_exact(key)
            if rec is not None:
                self._l1_hits += 1
                return LookupResult(
                    value=rec.value,
                    similarity=1.0,
                    matched_key=rec.key,
                    tier="L1_EXACT",
                    ttl=rec.ttl,
                    tags=rec.tags,
                )

            # 2. Check L2 Exact Hit in O(1)
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
                    ttl=rec.ttl,
                    tags=rec.tags,
                )

            # -----------------------------------------------------------------
            # SEMANTIC PATH: Vector Similarity Scan
            # -----------------------------------------------------------------
            # Only compute vector when exact match is a miss
            if query_vector is None and embed_fn is not None:
                query_vector = embed_fn(key)

            if query_vector is not None:
                # 3. Check L1 Semantic Hit in O(N*d)
                hit = self.l1.find_semantic(query_vector, self.config.similarity_threshold)
                if hit is not None:
                    sem_rec, score = hit
                    self._l1_hits += 1
                    return LookupResult(
                        value=sem_rec.value,
                        similarity=score,
                        matched_key=sem_rec.key,
                        tier="L1_SEMANTIC",
                        ttl=sem_rec.ttl,
                        tags=sem_rec.tags,
                    )

                # 4. Check L2 Semantic Hit in O(M*d)
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
                        ttl=sem_rec.ttl,
                        tags=sem_rec.tags,
                    )

            # 5. Missed everywhere
            self._misses += 1
            return None

    def set(
        self,
        key: str,
        value: str,
        vector: np.ndarray,
        ttl: Optional[int] = None,
        tags: Sequence[str] = (),
    ) -> None:
        """Store an answer in L1 RAM with optional TTL and categorization tags."""
        with self._lock:
            # Calculate expiration timestamp
            effective_ttl = ttl if ttl is not None else self.config.default_ttl
            expires_at = (time.time() + effective_ttl) if (effective_ttl is not None and effective_ttl > 0) else None

            # Update tag index
            self._update_tags(key, tags)

            # If key was already in L2, remove it so it only lives in L1 (strict tiering)
            if key in self.l2:
                self.l2.remove(key)

            evicted = self.l1.put(key, value, vector, expires_at=expires_at, tags=tags)

            # If an item was pushed off the desk, save it safely in the filing cabinet
            if evicted is not None:
                self._evictions += 1
                self.l2.append(
                    evicted.key,
                    evicted.value,
                    evicted.vector,
                    expires_at=evicted.expires_at,
                    tags=evicted.tags,
                )

    def _update_tags(self, key: str, tags: Sequence[str]) -> None:
        """Update tag indices for a given key."""
        old_tags = self._key_to_tags.get(key, set())
        for ot in old_tags:
            self._tag_to_keys[ot].discard(key)
            if not self._tag_to_keys[ot]:
                self._tag_to_keys.pop(ot, None)

        if tags:
            tag_set = set(tags)
            self._key_to_tags[key] = tag_set
            for t in tag_set:
                self._tag_to_keys[t].add(key)
        else:
            self._key_to_tags.pop(key, None)

    def _promote_to_l1(self, record: CacheRecord) -> None:
        """Move a cold item from L2 Disk back onto the warm L1 RAM desk (strict tiering)."""
        self.l2.remove(record.key)
        evicted = self.l1.put(
            record.key,
            record.value,
            record.vector,
            expires_at=record.expires_at,
            tags=record.tags,
        )
        if evicted is not None:
            self._evictions += 1
            self.l2.append(
                evicted.key,
                evicted.value,
                evicted.vector,
                expires_at=evicted.expires_at,
                tags=evicted.tags,
            )

    def expire(self, key: str, ttl_seconds: float) -> bool:
        """Set a time-to-live on an existing key across L1 or L2 in seconds."""
        with self._lock:
            # Try L1
            if key in self.l1:
                return self.l1.expire(key, ttl_seconds)
            # Try L2
            if key in self.l2:
                rec = self.l2.get_exact(key)
                if rec is not None:
                    # Append updated record with new TTL
                    new_exp = time.time() + ttl_seconds
                    self.l2.append(rec.key, rec.value, rec.vector, expires_at=new_exp, tags=rec.tags)
                    return True
            return False

    def ttl(self, key: str) -> int:
        """Return remaining TTL for key (-2 if missing, -1 if no TTL, >=0 remaining)."""
        with self._lock:
            if key in self.l1:
                return self.l1.ttl(key)
            if key in self.l2:
                rec = self.l2.get_exact(key)
                if rec is not None:
                    return rec.ttl
            return -2

    def invalidate_tag(self, tag: str) -> int:
        """Delete all cached items associated with a given tag in O(tagged_keys)."""
        with self._lock:
            keys = list(self._tag_to_keys.get(tag, set()))
            count = 0
            for k in keys:
                if self.delete(k):
                    count += 1
            return count

    def sweep_expired(self) -> int:
        """Active expiration sweep across both tiers."""
        with self._lock:
            swept_l1 = self.l1.sweep_expired()
            swept_l2 = self.l2.sweep_expired()
            total = swept_l1 + swept_l2
            self._expired_purges += total
            return total

    def delete(self, key: str) -> bool:
        """Delete an item from L1 and/or L2 in strict O(1) time."""
        with self._lock:
            self._update_tags(key, ())
            del_l1 = self.l1.delete(key)
            del_l2 = self.l2.remove(key)
            return del_l1 or del_l2

    def compact(self) -> int:
        """Compact L2 disk storage by purging deleted, overwritten, and expired records."""
        with self._lock:
            return self.l2.compact()

    def clear(self) -> None:
        """Clear all cached records in both L1 RAM and L2 Disk."""
        with self._lock:
            self.l1.clear()
            self.l2.clear()
            self._tag_to_keys.clear()
            self._key_to_tags.clear()

    def stats(self) -> dict[str, Any]:
        """Return system metrics: counts, hits, misses, evictions, and expired purges."""
        with self._lock:
            return {
                "l1_count": len(self.l1),
                "l2_count": len(self.l2),
                "total_count": len(self.l1) + len(self.l2),
                "l1_capacity": self.config.ram_capacity,
                "l1_hits": self._l1_hits,
                "l2_hits": self._l2_hits,
                "total_hits": self._l1_hits + self._l2_hits,
                "misses": self._misses,
                "evictions": self._evictions,
                "expired_purges": self._expired_purges,
                "active_tags_count": len(self._tag_to_keys),
            }

    def __len__(self) -> int:
        """Return total unique items across L1 and L2."""
        with self._lock:
            return len(self.l1) + len(self.l2)

    def __contains__(self, key: str) -> bool:
        """Check membership across L1 and L2 in O(1) time."""
        with self._lock:
            return (key in self.l1) or (key in self.l2)

    def close(self) -> None:
        """Close storage handles cleanly and stop background sweeper."""
        self._stop_sweep.set()
        if self._sweep_thread is not None and self._sweep_thread.is_alive():
            self._sweep_thread.join(timeout=1.0)
        with self._lock:
            self.l2.close()
