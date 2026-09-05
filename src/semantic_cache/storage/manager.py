"""
Tiered Semantic Cache - Storage Manager (The "Office Boss")
===========================================================

What is this file?
------------------
This file coordinates our two-tier storage system:
- L1 RAM: Fast Office Desk (holds items in quick memory).
- L2 Disk: Metal Filing Cabinet (saves older items to the hard drive).

How the Boss Handles Lookups (The Super-Fast Hierarchy):
--------------------------------------------------------
1. Fast-Path Exact Lookups (Bypasses vector math entirely):
   - Step 1: Check the Desk (L1 RAM) for an exact match. (Instant ~1 microsecond!)
   - Step 2: Check the Filing Cabinet (L2 Disk) for an exact match.
             If found on disk, the boss promotes it back onto the warm desk.
   - If an exact match is found, we NEVER waste time converting text to an arrow!

2. Semantic Meaning Scan (When exact match misses):
   - Step 3: Turn your question into an arrow vector.
   - Step 4: Multiply against arrows on the Desk (L1).
   - Step 5: Multiply against arrows in the Cabinet (L2).
   - If found in the cabinet, promote it back onto the warm desk.

Thread Safety & Background Cleanup:
-----------------------------------
- Lock Protection: All actions use an RLock so multiple customer threads can't clash.
- Active Sweeper: A quiet background worker thread wakes up periodically to toss
  out expired answers automatically.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Callable, Optional, Sequence, Tuple
import numpy as np

logger = logging.getLogger("semantic_cache.storage")

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
    """Boss coordinating L1 in-memory cache and L2 persistent disk storage."""

    def __init__(self, config: CacheConfig) -> None:
        """Initialize both storage tiers using system settings."""
        self.config = config
        self.l1 = L1RAMCache(capacity=config.ram_capacity, dim=config.vector_dim)
        self.l2 = L2DiskCache(
            file_path=config.disk_path,
            dim=config.vector_dim,
            enable_index_file=config.enable_index_file,
        )
        self._lock = threading.RLock()

        # Tag index folders: tag -> set(keys) and key -> set(tags)
        self._tag_to_keys: dict[str, set[str]] = defaultdict(set)
        self._key_to_tags: dict[str, set[str]] = defaultdict(set)

        # Performance score counters
        self._l1_hits: int = 0
        self._l2_hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._expired_purges: int = 0

        # Background active expiration sweeper thread
        self._stop_sweep = threading.Event()
        self._sweep_thread: Optional[threading.Thread] = None
        if self.config.enable_active_sweep:
            self._sweep_thread = threading.Thread(
                target=self._active_sweep_loop,
                name="tsc-expiry-sweeper",
                daemon=True,
            )
            self._sweep_thread.start()
            logger.debug("Active sweep thread started (interval=%.1fs)", self.config.sweep_interval_sec)

    def _active_sweep_loop(self) -> None:
        """Background helper that wakes up periodically to toss out expired records."""
        while not self._stop_sweep.wait(timeout=self.config.sweep_interval_sec):
            try:
                swept = self.sweep_expired()
                if swept > 0:
                    logger.debug("Sweep cleaned %d expired items", swept)
            except Exception:
                logger.warning("Sweep cycle failed", exc_info=True)

    def get(
        self,
        key: str,
        embed_fn: Optional[Callable[[str], np.ndarray]] = None,
        query_vector: Optional[np.ndarray] = None,
        key_prefix: Optional[str] = None,
    ) -> Optional[LookupResult]:
        """Search the cache hierarchy: L1 Exact -> L2 Exact -> L1 Semantic -> L2 Semantic.

        Fast-Path Optimization:
        Checks L1 and L2 for exact matches first. Only converts text into
        an arrow vector when both exact checks have missed!

        Args:
            key_prefix: If set, semantic search only considers keys starting
                        with this prefix (prevents cross-tenant matches).
        """
        with self._lock:
            # -----------------------------------------------------------------
            # FAST PATH: 1-Step Exact Lookups (Zero vector math needed!)
            # -----------------------------------------------------------------
            # 1. Check Desk (L1 RAM)
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

            # 2. Check Filing Cabinet (L2 Disk)
            rec = self.l2.get_exact(key)
            if rec is not None:
                self._l2_hits += 1
                # Promote from cabinet back onto desk!
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
            # SEMANTIC PATH: Meaning Search by Comparing Arrows
            # -----------------------------------------------------------------
            if query_vector is None and embed_fn is not None:
                query_vector = embed_fn(key)

            if query_vector is not None:
                # 3. Check Desk for meaning match
                hit = self.l1.find_semantic(query_vector, self.config.similarity_threshold, key_prefix=key_prefix)
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

                # 4. Check Filing Cabinet for meaning match
                hit = self.l2.find_semantic(query_vector, self.config.similarity_threshold, key_prefix=key_prefix)
                if hit is not None:
                    sem_rec, score = hit
                    self._l2_hits += 1
                    # Promote from cabinet back onto desk!
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
            logger.debug("Cache miss: key=%s", key[:80])
            return None

    def set(
        self,
        key: str,
        value: str,
        vector: np.ndarray,
        ttl: Optional[int] = None,
        tags: Sequence[str] = (),
    ) -> None:
        """Store an answer on the desk (L1) with optional expiration timer and tags."""
        with self._lock:
            effective_ttl = ttl if ttl is not None else self.config.default_ttl
            expires_at = (time.time() + effective_ttl) if (effective_ttl is not None and effective_ttl > 0) else None

            self._update_tags(key, tags)

            # Strict single copy: if it was in the cabinet, remove it so it's only on the desk
            if key in self.l2:
                self.l2.remove(key)

            evicted = self.l1.put(key, value, vector, expires_at=expires_at, tags=tags)

            # If an item slid off the desk, save it safely in the filing cabinet
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
        """Keep tag index up-to-date for fast group invalidation."""
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
        """Promote a cold item from the filing cabinet back onto the warm desk."""
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
        """Set or update an expiration timer on an existing item in seconds."""
        with self._lock:
            if key in self.l1:
                return self.l1.expire(key, ttl_seconds)
            if key in self.l2:
                rec = self.l2.get_exact(key)
                if rec is not None:
                    new_exp = time.time() + ttl_seconds
                    self.l2.append(rec.key, rec.value, rec.vector, expires_at=new_exp, tags=rec.tags)
                    return True
            return False

    def ttl(self, key: str) -> int:
        """Check remaining seconds before an item expires (-2 if missing, -1 if no timer)."""
        with self._lock:
            if key in self.l1:
                return self.l1.ttl(key)
            if key in self.l2:
                rec = self.l2.get_exact(key)
                if rec is not None:
                    return rec.ttl
            return -2

    def invalidate_tag(self, tag: str) -> int:
        """Delete all cached items labeled with a specific tag (batch single-pass).

        TC-2 Optimization: Instead of calling delete() N times (each of which
        re-iterates the tag index), we directly pop the tag and do one pass.
        """
        with self._lock:
            keys = list(self._tag_to_keys.pop(tag, set()))
            if not keys:
                return 0

            count = 0
            for k in keys:
                # Remove only THIS tag from the key's tag set (skip full _update_tags)
                key_tags = self._key_to_tags.get(k)
                if key_tags:
                    key_tags.discard(tag)
                    if not key_tags:
                        self._key_to_tags.pop(k, None)

                # Remove from both tiers directly
                del_l1 = self.l1.delete(k)
                del_l2 = self.l2.remove(k)
                if del_l1 or del_l2:
                    count += 1

            logger.debug("Batch tag invalidation: tag=%s, removed=%d keys", tag, count)
            return count

    def sweep_expired(self) -> int:
        """Toss out all expired items across both the desk and the cabinet.

        SEC-5: Wraps each tier's sweep in error handling so an issue in one tier
        does not prevent the other tier from cleaning up.
        PR-6: Automatically triggers disk compaction if dead space from TTL updates
        exceeds the configured waste ratio threshold.
        """
        with self._lock:
            total = 0
            try:
                swept_l1 = self.l1.sweep_expired()
                total += swept_l1
            except Exception as e:
                logger.error("L1 sweep_expired failed: %s", e, exc_info=True)

            try:
                swept_l2 = self.l2.sweep_expired()
                total += swept_l2
            except Exception as e:
                logger.error("L2 sweep_expired failed: %s", e, exc_info=True)

            self._expired_purges += total

            # PR-6: Auto-compact if wasted disk space exceeds threshold
            if self.config.auto_compact_waste_ratio > 0:
                try:
                    total_size, wasted, ratio = self.l2.waste_stats()
                    # Only trigger auto-compact if waste is non-trivial (>64KB) and exceeds threshold
                    if wasted > 65536 and ratio >= self.config.auto_compact_waste_ratio:
                        logger.info(
                            "Auto-compacting L2 disk: waste ratio %.1f%% (%d/%d bytes)",
                            ratio * 100.0,
                            wasted,
                            total_size,
                        )
                        self.l2.compact()
                except Exception as e:
                    logger.warning("Auto-compact check failed: %s", e, exc_info=True)

            return total

    def delete(self, key: str) -> bool:
        """Delete an answer completely in 1 instant step."""
        with self._lock:
            self._update_tags(key, ())
            del_l1 = self.l1.delete(key)
            del_l2 = self.l2.remove(key)
            return del_l1 or del_l2

    def compact(self) -> int:
        """Clean up the filing cabinet file on disk to free up wasted space.

        LAT-5 Note: This holds the global lock during file rewrite.
        Since _execute_command runs in a thread pool (LAT-1 fix), this
        won't block the event loop, but it will block other cache ops.
        """
        with self._lock:
            start = time.time()
            reclaimed = self.l2.compact()
            elapsed_ms = (time.time() - start) * 1000
            if elapsed_ms > 100:
                logger.warning("compact() took %.1fms (held lock the entire time)", elapsed_ms)
            return reclaimed

    def clear(self) -> None:
        """Wipe both desk and filing cabinet clean."""
        with self._lock:
            self.l1.clear()
            self.l2.clear()
            self._tag_to_keys.clear()
            self._key_to_tags.clear()

    def stats(self) -> dict[str, Any]:
        """Return system health metrics: item counts, hits, misses, and evictions."""
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
        """Total number of unique items across both tiers."""
        with self._lock:
            return len(self.l1) + len(self.l2)

    def __contains__(self, key: str) -> bool:
        """Check if an item exists anywhere in the cache."""
        with self._lock:
            return (key in self.l1) or (key in self.l2)

    def close(self) -> None:
        """Cleanly close files and stop the background cleaner thread."""
        self._stop_sweep.set()
        if self._sweep_thread is not None and self._sweep_thread.is_alive():
            self._sweep_thread.join(timeout=1.0)
        with self._lock:
            self.l2.close()
