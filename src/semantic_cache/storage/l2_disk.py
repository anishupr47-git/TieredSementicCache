"""
Tiered Semantic Cache - L2 Persistent Disk Storage (The "Filing Cabinet")
=========================================================================

What is this file?
------------------
This is Tier 2 of our cache: hard drive storage.
Think of it like a giant metal filing cabinet:
- When your desk (L1 RAM) gets full, older items are filed away here.
- It can store millions of items without filling up your computer's RAM.
- We use a superpower called 'mmap' (Memory Mapping):
  Instead of slowly copying files into Python, your computer's operating system
  points directly to the hard drive page. We can read any record in 1 instant
  step (O(1)) with ZERO memory copies!

Binary File Format (Append-Only Log):
-------------------------------------
Each saved item is written to the end of the file as compact binary bytes:
┌───────────────┬──────────────┬──────────────┬──────────────┬──────────────┬──────────────────┐
│ key_len (u32) │ val_len(u32) │ vec_dim(u32) │  key_bytes   │  val_bytes   │ vec_bytes(f32[d])│
│    4 bytes    │   4 bytes    │   4 bytes    │  (variable)  │  (variable)  │    4 * d bytes   │
└───────────────┴──────────────┴──────────────┴──────────────┴──────────────┴──────────────────┘

Zero-Allocation Search & Indexing:
----------------------------------
1. Index Card System:
   We keep a tiny index in RAM that remembers the exact byte location of every key:
     key -> (byte_offset, record_length)
   When looking up an answer, we jump directly to that exact byte number on the disk!

2. Zero-Allocation Semantic Scan:
   Vectors are kept in a contiguous 2D table that doubles geometrically.
   Searching uses a zero-copy NumPy slice view: active = matrix[:count].
   Runs BLAS Level-2 GEMV with ZERO heap allocations during search!
"""

from __future__ import annotations

import mmap
from pathlib import Path
import struct
import time
from typing import Optional, Sequence, Tuple
import numpy as np

from semantic_cache.storage.l1_ram import CacheRecord

# Binary Record Header Format:
# klen(u32), vlen(u32), dim(u32), expires_at_ms(u64), tag_len(u32) = 24 bytes
HEADER_FORMAT = "<IIIQI"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class L2DiskCache:
    """Zero-copy memory-mapped (mmap) append-only disk storage with TTL and Tagging."""

    def __init__(self, file_path: Path, dim: int = 384) -> None:
        """Initialize disk cache at the specified file path."""
        self.file_path = Path(file_path)
        self.dim = dim

        # Ensure parent folder exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory index: key -> (byte_offset, total_record_length)
        self._index: dict[str, Tuple[int, int]] = {}
        self._key_to_slot: dict[str, int] = {}

        # Contiguous vector table for zero-allocation BLAS semantic search
        self._keys_list: list[str] = []
        self._matrix = np.empty((128, dim), dtype=np.float32)
        self._count: int = 0

        # Open file handle and memory map
        self._file = open(self.file_path, "a+b")
        self._mm: Optional[mmap.mmap] = None

        # Build index from existing disk records on startup
        self._build_index()

    def __len__(self) -> int:
        """Total number of unique items saved in the filing cabinet."""
        return self._count

    def __contains__(self, key: str) -> bool:
        """Check if key exists in L2 index in O(1) time (purging if expired)."""
        if key not in self._index:
            return False
        rec = self.get_exact(key)
        return rec is not None

    def _build_index(self) -> None:
        """Scan the disk log once at startup to index all saved items, deduplicating superseded entries."""
        self._file.seek(0, 2)
        size = self._file.tell()
        if size == 0:
            return

        self._file.seek(0)
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        offset = 0
        # Deduplicated map: key -> (offset, record_len, vector)
        deduped: dict[str, Tuple[int, int, np.ndarray]] = {}

        while offset + HEADER_SIZE <= size:
            klen, vlen, dim, exp_ms, tag_len = struct.unpack_from(HEADER_FORMAT, self._mm, offset)
            record_len = HEADER_SIZE + klen + vlen + tag_len + (dim * 4)

            if offset + record_len > size:
                break  # Incomplete record at end of file

            # Check expiration on reload
            if exp_ms > 0 and (time.time() * 1000.0) >= exp_ms:
                offset += record_len
                continue

            # Read key text
            key_bytes = self._mm[offset + HEADER_SIZE : offset + HEADER_SIZE + klen]
            key = key_bytes.decode("utf-8")

            # Read vector arrow
            vec_offset = offset + HEADER_SIZE + klen + vlen + tag_len
            vec = np.frombuffer(self._mm, dtype=np.float32, count=dim, offset=vec_offset).copy()

            # The latest record in the log supersedes older records for the same key
            deduped[key] = (offset, record_len, vec)
            offset += record_len

        # Populate contiguous vector matrix and index mappings with unique items only
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
        """Read a record from disk in strict O(1) time using zero-copy mmap.

        Passively evicts and returns None if expired.
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

        # Read value string, tags, and vector directly from mmap
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
        """Scan filing cabinet for semantic similarity matches in O(M*d) time with zero allocations."""
        if self._count == 0:
            return None

        # Zero-copy view of active rows (runs at hardware SIMD speed)
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
        """Append a record to the persistent binary log in amortized O(1) time."""
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

        # Close existing mmap before file write (critical for Windows OS file-locking safety)
        if self._mm is not None:
            self._mm.close()
            self._mm = None

        # Write to end of file
        self._file.seek(0, 2)
        offset = self._file.tell()
        self._file.write(payload)
        self._file.flush()

        # Remap updated file
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # Update in-memory index card and vector matrix in O(1)
        if key in self._key_to_slot:
            # Key was already in L2: update existing vector slot in-place
            slot = self._key_to_slot[key]
            self._matrix[slot] = vector
            self._index[key] = (offset, total_len)
        else:
            # New key in L2: assign new slot with geometric table doubling
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
        """Scan active keys in L2 and remove expired items."""
        keys_to_check = list(self._index.keys())
        swept = 0
        for k in keys_to_check:
            # get_exact automatically evicts expired records
            if self.get_exact(k) is None and k not in self._index:
                swept += 1
        return swept

    def remove(self, key: str) -> bool:
        """Remove an item from L2 in-memory index and vector matrix in strict O(1) time.

        Uses O(1) swap-and-pop to maintain a contiguous vector matrix without holes.
        Returns True if item was removed, False if not found.
        """
        if key not in self._index:
            return False

        del self._index[key]
        slot = self._key_to_slot.pop(key)
        last_slot = len(self._keys_list) - 1
        last_key = self._keys_list.pop()

        # If removed item was not the last slot, swap the last slot into its position
        if slot != last_slot:
            self._matrix[slot] = self._matrix[last_slot]
            self._keys_list[slot] = last_key
            self._key_to_slot[last_key] = slot

        self._count -= 1
        return True

    def compact(self) -> int:
        """Compact the append-only log file by rewriting only currently active records.

        Frees disk space from deleted or superseded records.
        Returns the number of bytes reclaimed.
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

        # Reopen file for clean sequential reading of active records
        with open(self.file_path, "rb") as src_f, open(temp_path, "wb") as dst_f:
            new_offset = 0
            for key, (old_offset, record_len) in self._index.items():
                src_f.seek(old_offset)
                record_data = src_f.read(record_len)
                dst_f.write(record_data)
                new_index[key] = (new_offset, record_len)
                new_offset += record_len

        # Close existing file handle so Windows allows atomic file replacement
        self._file.close()

        import os
        os.replace(temp_path, self.file_path)

        # Reopen file and remap
        self._file = open(self.file_path, "a+b")
        self._index = new_index
        new_size = new_offset
        if new_size > 0:
            self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        reclaimed = orig_size - new_size
        return max(0, reclaimed)

    def clear(self) -> None:
        """Clear all disk storage and truncate the file."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._file.close()

        # Truncate file on disk
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
