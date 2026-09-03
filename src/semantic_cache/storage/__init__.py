"""
Tiered Semantic Cache - Storage Engine Package
==============================================

Exports the two-tier storage system:
- CacheRecord: In-memory data container.
- L1RAMCache: Strict O(1) in-memory LRU cache.
- L2DiskCache: Append-only persistent binary disk log with zero-copy mmap reads.
- StorageManager: Two-tier coordinator handling L1 eviction and L2 promotion.
"""

from semantic_cache.storage.l1_ram import CacheRecord, L1RAMCache
from semantic_cache.storage.l2_disk import L2DiskCache
from semantic_cache.storage.manager import LookupResult, StorageManager

__all__ = [
    "CacheRecord",
    "L1RAMCache",
    "L2DiskCache",
    "LookupResult",
    "StorageManager",
]
