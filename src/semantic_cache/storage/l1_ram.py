"""
Tiered Semantic Cache - L1 In-Memory RAM Cache (The "Clean Desk")
================================================================

What is this file?
------------------
This is Tier 1 of our cache: fast computer memory (RAM).
Think of it like the top of your office desk:
- You keep the things you use most often right in front of you.
- Your desk has limited space (ram_capacity).
- If your desk gets full, the paper you haven't touched for the longest time
  (Least Recently Used, or LRU) slides off your desk into the filing cabinet (L2 Disk).

How it stays super fast (Low-Latency Math & Logic in Points):
-------------------------------------------------------------
1. Strict O(1) Instant Exact Search:
   - Uses Python's OrderedDict (a hash map + doubly linked list built in C).
   - Finding any exact text takes 1 quick step (O(1)).
   - When an item is read, it moves to the top of the stack in 1 step:
     move_to_end(key, last=True).

2. Strict O(1) Eviction:
   - When the desk is full, popitem(last=False) removes the oldest item
     from the bottom in exactly 1 step (O(1)).

3. Zero-Allocation Semantic Search:
   - All saved vectors are kept in a contiguous 2D table (matrix).
   - When removing or evicting an item, we swap its row with the last row
     in the matrix (slot swap in O(1)).
   - When searching, we multiply all rows at once using hardware SIMD (BLAS).
     Zero memory allocations during query time!
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass
class CacheRecord:
    """A single cached item holding text, answer, and its direction arrow."""

    key: str
    value: str
    vector: np.ndarray


class L1RAMCache:
    """Strict O(1) LRU in-memory cache with hardware-accelerated vector search."""

    def __init__(self, capacity: int = 1000, dim: int = 384) -> None:
        """Initialize L1 cache with maximum desk space and arrow dimension."""
        if capacity <= 0:
            raise ValueError(f"Capacity must be > 0, got {capacity}")
        if dim <= 0:
            raise ValueError(f"Dimension must be > 0, got {dim}")

        self.capacity = capacity
        self.dim = dim

        # 1. Doubly linked hash map for strict O(1) exact lookup & LRU order
        self._records: OrderedDict[str, CacheRecord] = OrderedDict()

        # 2. Contiguous 2D matrix for instant BLAS vector search
        self._matrix = np.zeros((capacity, dim), dtype=np.float32)
        self._matrix_keys: list[str] = []
        self._key_to_slot: dict[str, int] = {}

    def __len__(self) -> int:
        """Current number of items on the desk."""
        return len(self._records)

    def is_full(self) -> bool:
        """True if desk has reached maximum capacity."""
        return len(self._records) >= self.capacity

    def get_exact(self, key: str) -> Optional[CacheRecord]:
        """Look up by exact text in O(1) time.

        If found, marks the item as 'most recently used' by moving it to the top.
        """
        if key not in self._records:
            return None

        # O(1) move to top of recent list
        self._records.move_to_end(key, last=True)
        return self._records[key]

    def find_semantic(
        self,
        query_vector: np.ndarray,
        threshold: float,
    ) -> Optional[Tuple[CacheRecord, float]]:
        """Find the closest saved answer using direction arrows in O(N*d) time.

        Simple steps:
        1. Multiply all saved arrows against query arrow at once (BLAS).
        2. Find the highest score (np.argmax).
        3. If score >= threshold, move item to top of desk and return it!
        """
        count = len(self._matrix_keys)
        if count == 0:
            return None

        # Slicing up to count is a zero-copy NumPy view (nanosecond speed)
        active_matrix = self._matrix[:count]
        scores = np.dot(active_matrix, query_vector)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            best_key = self._matrix_keys[best_idx]
            # Mark as recently used in O(1)
            self._records.move_to_end(best_key, last=True)
            return self._records[best_key], best_score

        return None

    def put(self, key: str, value: str, vector: np.ndarray) -> Optional[CacheRecord]:
        """Put a new item on the desk.

        If the desk is full, kicks out the oldest item (LRU) in O(1) time
        and returns it so it can be saved to disk.
        """
        evicted: Optional[CacheRecord] = None

        # If key already exists, update it cleanly
        if key in self._records:
            self._remove_from_matrix(key)
            self._records.pop(key)
        elif self.is_full():
            # Desk full! Eject the oldest item (front of OrderedDict) in O(1)
            oldest_key, oldest_record = self._records.popitem(last=False)
            self._remove_from_matrix(oldest_key)
            evicted = oldest_record

        # Store in records
        rec = CacheRecord(key=key, value=value, vector=vector)
        self._records[key] = rec

        # Store in vector matrix slot in O(1)
        slot = len(self._matrix_keys)
        self._matrix[slot] = vector
        self._matrix_keys.append(key)
        self._key_to_slot[key] = slot

        return evicted

    def _remove_from_matrix(self, key: str) -> None:
        """Remove an item's vector using O(1) slot swap (swap with last element)."""
        slot = self._key_to_slot.pop(key)
        last_slot = len(self._matrix_keys) - 1
        last_key = self._matrix_keys.pop()

        # If removed item was not already at the end, swap last item into its slot
        if slot != last_slot:
            self._matrix[slot] = self._matrix[last_slot]
            self._matrix_keys[slot] = last_key
            self._key_to_slot[last_key] = slot
