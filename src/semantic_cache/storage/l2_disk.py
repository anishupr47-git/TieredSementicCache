"""
Tiered Semantic Cache - L2 Persistent Disk Storage (The "Filing Cabinet")
=========================================================================

What is this file?
------------------
This is Tier 2 of our cache: hard drive storage.
Think of it like a giant metal filing cabinet:
- When your desk (L1 RAM) gets full, older items are filed away here.
- It can hold millions of answers without filling up your computer's memory.
- It uses a superpower called 'mmap' (Memory Mapping):
  Instead of slowly loading whole files into memory, your computer reads directly
  off the hard drive like pointing a finger at a page in a book. It can read
  any record instantly with ZERO copying!

Binary File Format (Append-Only Log):
-------------------------------------
Each saved item is written to the end of the file as compact binary bytes:
┌───────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────────┐
│ key_len (u32) │ val_len(u32) │ vec_dim(u32) │ expires_time │ tag_len(u32) │     payload      │
│    4 bytes    │   4 bytes    │   4 bytes    │   8 bytes    │   4 bytes    │  (text + vector) │
└───────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────────┘

Simple Math & Logic in Points:
------------------------------
1. Instant Index Card System:
   - We keep a tiny index in memory remembering the exact byte number of every key.
   - To read an answer, we jump straight to that byte on the hard drive in 1 step!

2. Fast Arrow Scanning:
   - All saved direction arrows are kept in a clean table for instant comparison.
   - When searching, we multiply your question's arrow against all saved arrows at once.

3. Log Compaction (Taking Out the Trash):
   - Over time, updated or deleted answers leave dead space in the log file.
   - 'compact()' rewrites only active, living answers to a fresh file, reclaiming disk space!
"""

from __future__ import annotations

import mmap
import os
from pathlib import Path
import struct
import time
from typing import Optional, Sequence, Tuple
import numpy as np

from semantic_cache.storage.l1_ram import CacheRecord

# Binary Record Header Format:
# klen(u32), vlen(u32), dim(u32), expires_at_ms(u64), tag_len(u32) = 24 bytes total
HEADER_FORMAT = "<IIIQI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class L2DiskCache:
    """Zero-copy memory-mapped append-only disk storage with TTL, tags, and compaction."""

    def __init__(self, file_path: Path, dim: int = 384) -> None:
        """Set up filing cabinet at the given file path."""
        self.file_path = Path(file_path)
        self.dim = dim

        # Make sure storage folder exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Quick index card map: key -> (byte_offset, total_record_length)
        self._index: dict[str, Tuple[int, int]] = {}
        self._key_to_slot: dict[str, int] = {}

        # Contiguous table of arrows for fast vector matching
        self._keys_list: list[str] = []
        self._matrix = np.empty((128, dim), dtype=np.float32)
        self._count: int = 0

        # Open file handle and memory map
        self._file = open(self.file_path, "a+b")
        self._mm: Optional[mmap.mmap] = None

        # Scan existing records from previous sessions
        self._build_index()

    def __len__(self) -> int:
        """Count total unique answers in the filing cabinet."""
        return self._count

    def __contains__(self, key: str) -> bool:
        """Check if an answer is in the cabinet (removes if expired)."""
        if key not in self._index:
            return False
        rec = self.get_exact(key)
        return rec is not None

    def _build_index(self) -> None:
        """Scan the disk file on startup to index all saved items.

        Automatically skips expired items and ignores older duplicate versions.
        """
        self._file.seek(0, 2)
        size = self._file.tell()
        if size == 0:
            return

        self._file.seek(0)
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        offset = 0
        deduped: dict[str, Tuple[int, int, np.ndarray]] = {}

        while offset + HEADER_SIZE <= size:
            klen, vlen, dim, exp_ms, tag_len = struct.unpack_from(HEADER_FORMAT, self._mm, offset)
            record_len = HEADER_SIZE + klen + vlen + tag_len + (dim * 4)

            if offset + record_len > size:
                break  # Incomplete record at the end of the file

            # Skip expired entries
            if exp_ms > 0 and (time.time() * 1000.0) >= exp_ms:
                offset += record_len
                continue

            # Read key text
            key_bytes = self._mm[offset + HEADER_SIZE : offset + HEADER_SIZE + klen]
            key = key_bytes.decode("utf-8")

            # Read arrow vector
            vec_offset = offset + HEADER_SIZE + klen + vlen + tag_len
            vec = np.frombuffer(self._mm, dtype=np.float32, count=dim, offset=vec_offset).copy()

            # Latest version always wins
            deduped[key] = (offset, record_len, vec)
            offset += record_len

        # Fill table with clean, unique items
        self._count = len(deduped)
        if self._count > 0:
            alloc_cap = max(128, self._count * 2)
            self._matrix = np.empty((alloc_cap, self.dim), dtype=np.float32)
            for slot, (k, (off, rlen, vec)) in enumerate(deduped.items()):
                self._index[k] = (off, rlen)
                self._keys_list.append(k)
                self._key_to_slot[k] = slot
                self._matrix[slot] = vec

    def get_exact(self, key: str) -> Optional[CacheRecord]:
        """Read an answer from disk in 1 instant step using zero-copy mmap.

        If expired, quietly removes it and returns None.
        """
        if key not in self._index:
            return None

        if self._mm is None:
            self._file.seek(0, 2)
            if self._file.tell() == 0:
                return None
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        offset, _ = self._index[key]
        klen, vlen, dim, exp_ms, tag_len = struct.unpack_from(HEADER_FORMAT, self._mm, offset)

        expires_at = (exp_ms / 1000.0) if exp_ms > 0 else None
        if expires_at is not None and time.time() >= expires_at:
            self.remove(key)
            return None

        # Read value string, tags, and vector directly from the memory map
        val_start = offset + HEADER_SIZE + klen
        val = self._mm[val_start : val_start + vlen].decode("utf-8")

        tag_start = val_start + vlen
        tag_str = self._mm[tag_start : tag_start + tag_len].decode("utf-8")
        tags = tuple(tag_str.split(",")) if tag_len > 0 else ()

        vec_start = tag_start + tag_len
        vec = np.frombuffer(self._mm, dtype=np.float32, count=dim, offset=vec_start).copy()

        return CacheRecord(
            key=key,
            value=val,
            vector=vec,
            expires_at=expires_at,
            tags=tags,
        )

    def find_semantic(
        self,
        query_vector: np.ndarray,
        threshold: float,
    ) -> Optional[Tuple[CacheRecord, float]]:
        """Search disk entries for meaning similarity in 1 instant calculation."""
        if self._count == 0:
            return None

        # Hardware-accelerated dot product across all active arrows
        active_matrix = self._matrix[: self._count]
        scores = np.dot(active_matrix, query_vector)

        candidate_indices = np.argsort(-scores)

        for idx in candidate_indices:
            score = float(scores[idx])
            if score < threshold:
                break

            best_key = self._keys_list[idx]
            record = self.get_exact(best_key)
            if record is not None:
                return record, score

        return None

    def append(
        self,
        key: str,
        value: str,
        vector: np.ndarray,
        expires_at: Optional[float] = None,
        tags: Sequence[str] = (),
    ) -> None:
        """Write an answer to the end of the filing cabinet file."""
        key_bytes = key.encode("utf-8")
        val_bytes = value.encode("utf-8")
        tag_bytes = ",".join(tags).encode("utf-8") if tags else b""
        vec_bytes = vector.astype(np.float32).tobytes()

        klen = len(key_bytes)
        vlen = len(val_bytes)
        tag_len = len(tag_bytes)
        dim = len(vector)
        exp_ms = int(expires_at * 1000.0) if expires_at is not None and expires_at > 0 else 0

        header = struct.pack(HEADER_FORMAT, klen, vlen, dim, exp_ms, tag_len)
        payload = header + key_bytes + val_bytes + tag_bytes + vec_bytes
        total_len = len(payload)

        # Close existing mmap before writing (keeps Windows file locking safe)
        if self._mm is not None:
            self._mm.close()
            self._mm = None

        # Write bytes to the end of the file
        self._file.seek(0, 2)
        offset = self._file.tell()
        self._file.write(payload)
        self._file.flush()

        # Remap updated file
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # Update index card and vector slot
        if key in self._key_to_slot:
            slot = self._key_to_slot[key]
            self._matrix[slot] = vector
            self._index[key] = (offset, total_len)
        else:
            slot = self._count
            if slot >= len(self._matrix):
                new_cap = max(128, len(self._matrix) * 2)
                new_matrix = np.empty((new_cap, self.dim), dtype=np.float32)
                new_matrix[:slot] = self._matrix[:slot]
                self._matrix = new_matrix

            self._matrix[slot] = vector
            self._keys_list.append(key)
            self._key_to_slot[key] = slot
            self._index[key] = (offset, total_len)
            self._count += 1

    def sweep_expired(self) -> int:
        """Toss out all expired answers currently filed on disk."""
        keys_to_check = list(self._index.keys())
        swept = 0
        for k in keys_to_check:
            if self.get_exact(k) is None and k not in self._index:
                swept += 1
        return swept

    def remove(self, key: str) -> bool:
        """Remove an item from disk index and vector table using Card-Deck Swap."""
        if key not in self._index:
            return False

        del self._index[key]
        slot = self._key_to_slot.pop(key)
        last_slot = len(self._keys_list) - 1
        last_key = self._keys_list.pop()

        # Swap last element into vacated slot to prevent gaps
        if slot != last_slot:
            self._matrix[slot] = self._matrix[last_slot]
            self._keys_list[slot] = last_key
            self._key_to_slot[last_key] = slot

        self._count -= 1
        return True

    def compact(self) -> int:
        """Take out the trash: rewrite only live answers, reclaiming wasted disk space.

        Returns:
            Number of bytes reclaimed.
        """
        if self._mm is not None:
            self._mm.close()
            self._mm = None

        self._file.seek(0, 2)
        orig_size = self._file.tell()
        if orig_size == 0 or len(self._index) == 0:
            return 0

        temp_path = self.file_path.with_suffix(".compact")
        new_index: dict[str, Tuple[int, int]] = {}

        # Copy only active, living records to a fresh clean file
        with open(self.file_path, "rb") as src_f, open(temp_path, "wb") as dst_f:
            new_offset = 0
            for key, (old_offset, record_len) in self._index.items():
                src_f.seek(old_offset)
                record_data = src_f.read(record_len)
                dst_f.write(record_data)
                new_index[key] = (new_offset, record_len)
                new_offset += record_len

        # Close and replace original file cleanly
        self._file.close()
        os.replace(temp_path, self.file_path)

        # Reopen and remap the cleaned file
        self._file = open(self.file_path, "a+b")
        self._index = new_index
        new_size = new_offset
        if new_size > 0:
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        reclaimed = orig_size - new_size
        return max(0, reclaimed)

    def clear(self) -> None:
        """Wipe the filing cabinet completely clean."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._file.close()

        with open(self.file_path, "wb"):
            pass

        self._file = open(self.file_path, "a+b")
        self._index.clear()
        self._keys_list.clear()
        self._key_to_slot.clear()
        self._matrix = np.empty((128, self.dim), dtype=np.float32)
        self._count = 0

    def close(self) -> None:
        """Safely close open file handles and memory maps."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if not self._file.closed:
            self._file.close()
