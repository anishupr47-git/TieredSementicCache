"""
Tiered Semantic Cache - L1 In-Memory RAM Cache (The "Fast Office Desk")
=======================================================================

What is this file?
------------------
This is Tier 1 of our cache: lightning-fast computer memory (RAM).
Think of it like the top of your office desk:
- You keep the answers you need most often right in front of you.
- Your desk only has room for a certain number of papers (ram_capacity).
- When the desk is full, the paper you haven't touched for the longest time
  (Least Recently Used, or LRU) slides off the desk and into the filing cabinet (L2 Disk).

How it stays super fast (The Simple Math & Logic):
--------------------------------------------------
1. Instant Exact Search (O(1) Hash Map):
   - Finding exact text takes 1 instant step using Python's OrderedDict.
   - When you read a paper, it moves right to the top of your desk so it won't be thrown out.

2. Instant Eviction (O(1) Swap-and-Pop):
   - When the desk is full, the oldest paper at the bottom is popped in 1 step.
   - To remove a sentence's arrow from our table, we swap it with the very last arrow.
     (Just like taking the bottom card from a deck to fill an empty slot—takes 1 instant step!).

3. Fast Meaning Search (Matrix Dot Product):
   - All arrows are lined up in a neat table (matrix).
   - In 1 hardware step, your question's arrow is multiplied against all saved arrows
     to find the closest meaning.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import time
from typing import Optional, Sequence, Tuple
import numpy as np


@dataclass
class CacheRecord:
    """A single cached item holding question, answer, direction arrow, TTL, and tags."""

    key: str
    value: str
    vector: np.ndarray
    expires_at: Optional[float] = None
    tags: Tuple[str, ...] = ()

    @property
    def is_expired(self) -> bool:
        """True if the clock has passed the expiration time."""
        if self.expires_at is None or self.expires_at <= 0:
            return False
        return time.time() >= self.expires_at

    @property
    def ttl(self) -> int:
        """Remaining seconds before expiration (-1 means it lives forever, 0 means expired)."""
        if self.expires_at is None or self.expires_at <= 0:
            return -1
        remaining = int(self.expires_at - time.time())
        return max(0, remaining)


class L1RAMCache:
    """Fast in-memory cache with instant LRU eviction and vector search."""

    def __init__(self, capacity: int = 1000, dim: int = 384) -> None:
        """Set up desk with maximum item capacity and arrow size."""
        if capacity <= 0:
            raise ValueError(f"Capacity must be > 0, got {capacity}")
        if dim <= 0:
            raise ValueError(f"Dimension must be > 0, got {dim}")

        self.capacity = capacity
        self.dim = dim

        # 1. Fast lookup table that also remembers what was used most recently
        self._records: OrderedDict[str, CacheRecord] = OrderedDict()

        # 2. Neat table of numbers (matrix) for instant arrow math
        self._matrix = np.zeros((capacity, dim), dtype=np.float32)
        self._matrix_keys: list[str] = []
        self._key_to_slot: dict[str, int] = {}

    def __len__(self) -> int:
        """Count how many items are currently on the desk."""
        return len(self._records)

    def __contains__(self, key: str) -> bool:
        """Check if a question is on the desk (removes it if expired)."""
        if key not in self._records:
            return False
        if self._records[key].is_expired:
            self.delete(key)
            return False
        return True

    def is_full(self) -> bool:
        """True if the desk has reached its maximum item limit."""
        return len(self._records) >= self.capacity

    def get_exact(self, key: str) -> Optional[CacheRecord]:
        """Look up by exact text in 1 instant step.

        - If found and not expired, moves the item to the top of the desk (most recently used).
        - If expired, removes it immediately and returns None.
        """
        if key not in self._records:
            return None

        rec = self._records[key]
        if rec.is_expired:
            self.delete(key)
            return None

        # Move to the top of the pile
        self._records.move_to_end(key, last=True)
        return rec

    def find_semantic(
        self,
        query_vector: np.ndarray,
        threshold: float,
    ) -> Optional[Tuple[CacheRecord, float]]:
        """Find the closest matching answer by comparing arrow directions.

        Skips and removes any expired items automatically.
        """
        count = len(self._matrix_keys)
        if count == 0:
            return None

        # Multiply question arrow against all active arrows in 1 instant hardware step
        active_matrix = self._matrix[:count]
        scores = np.dot(active_matrix, query_vector)

        # Check candidates from highest score to lowest
        candidate_indices = np.argsort(-scores)

        for idx in candidate_indices:
            score = float(scores[idx])
            if score < threshold:
                break  # The rest are too low to be a match

            candidate_key = self._matrix_keys[idx]
            rec = self._records.get(candidate_key)
            if rec is None:
                continue

            if rec.is_expired:
                self.delete(candidate_key)
                continue

            # Valid match found! Mark as recently used and return it
            self._records.move_to_end(candidate_key, last=True)
            return rec, score

        return None

    def put(
        self,
        key: str,
        value: str,
        vector: np.ndarray,
        expires_at: Optional[float] = None,
        tags: Sequence[str] = (),
    ) -> Optional[CacheRecord]:
        """Place a new answer on the desk.

        If the desk is full, pushes the oldest unused paper off the desk
        so it can be safely stored in the filing cabinet (L2 Disk).
        """
        evicted: Optional[CacheRecord] = None

        # If question already exists, update it cleanly
        if key in self._records:
            self._remove_from_matrix(key)
            self._records.pop(key)
        elif self.is_full():
            # Desk full! Push the oldest item off the bottom in 1 instant step
            oldest_key, oldest_record = self._records.popitem(last=False)
            self._remove_from_matrix(oldest_key)
            evicted = oldest_record

        # Save record with expiration time and tags
        rec = CacheRecord(
            key=key,
            value=value,
            vector=vector,
            expires_at=expires_at,
            tags=tuple(tags),
        )
        self._records[key] = rec

        # Assign arrow to the next open row in our table
        slot = len(self._matrix_keys)
        self._matrix[slot] = vector
        self._matrix_keys.append(key)
        self._key_to_slot[key] = slot

        return evicted

    def expire(self, key: str, ttl_seconds: float) -> bool:
        """Set a countdown timer on an existing answer."""
        if key not in self._records:
            return False
        rec = self._records[key]
        if rec.is_expired:
            self.delete(key)
            return False
        rec.expires_at = time.time() + ttl_seconds
        return True

    def ttl(self, key: str) -> int:
        """Check how many seconds are left before an answer expires (-2 if missing, -1 if no timer)."""
        if key not in self._records:
            return -2
        rec = self._records[key]
        if rec.is_expired:
            self.delete(key)
            return -2
        return rec.ttl

    def sweep_expired(self) -> int:
        """Active cleanup: scan the desk and toss out all expired answers."""
        expired_keys = [k for k, r in self._records.items() if r.is_expired]
        for k in expired_keys:
            self.delete(k)
        return len(expired_keys)

    def delete(self, key: str) -> bool:
        """Remove an item from the desk in 1 instant step."""
        if key not in self._records:
            return False

        self._remove_from_matrix(key)
        del self._records[key]
        return True

    def clear(self) -> None:
        """Wipe the desk clean."""
        self._records.clear()
        self._matrix_keys.clear()
        self._key_to_slot.clear()
        self._matrix.fill(0.0)

    def _remove_from_matrix(self, key: str) -> None:
        """Remove an arrow using the Card-Deck Swap trick (O(1) swap-and-pop).

        Instead of slowly shifting all arrows over to fill the empty hole,
        we just grab the very last arrow in the table and place it in the empty spot!
        """
        slot = self._key_to_slot.pop(key)
        last_slot = len(self._matrix_keys) - 1
        last_key = self._matrix_keys.pop()

        # If the item wasn't already at the very end, move the last item into this slot
        if slot != last_slot:
            self._matrix[slot] = self._matrix[last_slot]
            self._matrix_keys[slot] = last_key
            self._key_to_slot[last_key] = slot
