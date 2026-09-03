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

Index Card System (In-Memory Offset Map):
-----------------------------------------
We keep a tiny index in RAM that remembers the exact byte location of every key:
  key -> (byte_offset, record_length)
When looking up an answer, we jump directly to that exact byte number on the disk!
"""

from __future__ import annotations

import mmap
from pathlib import Path
import struct
from typing import Optional, Tuple
import numpy as np

from semantic_cache.storage.l1_ram import CacheRecord

HEADER_FORMAT = "<III"  # 3 unsigned 32-bit integers = 12 bytes
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class L2DiskCache:
    """Zero-copy memory-mapped (mmap) append-only disk storage."""

    def __init__(self, file_path: Path, dim: int = 384) -> None:
        """Initialize disk cache at the specified file path."""
        self.file_path = Path(file_path)
        self.dim = dim

        # Ensure parent folder exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # In-memory index: key -> (byte_offset, total_record_length)
        self._index: dict[str, Tuple[int, int]] = {}

        # Vector table for semantic scanning in L2
        self._keys_list: list[str] = []
        self._vectors_list: list[np.ndarray] = []

        # Open file handle and memory map
        self._file = open(self.file_path, "a+b")
        self._mm: Optional[mmap.mmap] = None

        # Build index from existing disk records on startup
        self._build_index()

    def __len__(self) -> int:
        """Total number of items saved in the filing cabinet."""
        return len(self._index)

    def _build_index(self) -> None:
        """Scan the disk log once at startup to index all saved items."""
        self._file.seek(0, 2)
        size = self._file.tell()
        if size == 0:
            return

        self._file.seek(0)
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        offset = 0
        while offset + HEADER_SIZE <= size:
            klen, vlen, dim = struct.unpack_from(HEADER_FORMAT, self._mm, offset)
            record_len = HEADER_SIZE + klen + vlen + (dim * 4)

            if offset + record_len > size:
                break  # Incomplete record at end of file

            # Read key text
            key_bytes = bytes(self._mm[offset + HEADER_SIZE : offset + HEADER_SIZE + klen])
            key = key_bytes.decode("utf-8")

            # Read vector arrow
            vec_offset = offset + HEADER_SIZE + klen + vlen
            vec = np.frombuffer(self._mm, dtype=np.float32, count=dim, offset=vec_offset).copy()

            self._index[key] = (offset, record_len)
            self._keys_list.append(key)
            self._vectors_list.append(vec)

            offset += record_len

    def get_exact(self, key: str) -> Optional[CacheRecord]:
        """Read a record from disk in strict O(1) time using zero-copy mmap."""
        if key not in self._index or self._mm is None:
            return None

        offset, _ = self._index[key]
        klen, vlen, dim = struct.unpack_from(HEADER_FORMAT, self._mm, offset)

        # Read value string and vector directly from mmap
        val_start = offset + HEADER_SIZE + klen
        val_bytes = bytes(self._mm[val_start : val_start + vlen])
        val = val_bytes.decode("utf-8")

        vec_start = val_start + vlen
        vec = np.frombuffer(self._mm, dtype=np.float32, count=dim, offset=vec_start).copy()

        return CacheRecord(key=key, value=val, vector=vec)

    def find_semantic(
        self,
        query_vector: np.ndarray,
        threshold: float,
    ) -> Optional[Tuple[CacheRecord, float]]:
        """Scan filing cabinet for semantic similarity matches in O(M*d) time."""
        if not self._vectors_list:
            return None

        # Stack L2 vectors for instant BLAS matrix multiplication
        matrix = np.array(self._vectors_list, dtype=np.float32)
        scores = np.dot(matrix, query_vector)

        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= threshold:
            best_key = self._keys_list[best_idx]
            record = self.get_exact(best_key)
            if record is not None:
                return record, best_score

        return None

    def append(self, key: str, value: str, vector: np.ndarray) -> None:
        """Append a new record to the end of the binary file log in O(1) time."""
        key_bytes = key.encode("utf-8")
        val_bytes = value.encode("utf-8")
        vec_bytes = vector.astype(np.float32).tobytes()

        klen = len(key_bytes)
        vlen = len(val_bytes)
        dim = len(vector)

        header = struct.pack(HEADER_FORMAT, klen, vlen, dim)
        payload = header + key_bytes + val_bytes + vec_bytes
        total_len = len(payload)

        # Write to end of file
        self._file.seek(0, 2)
        offset = self._file.tell()
        self._file.write(payload)
        self._file.flush()

        # Safely refresh mmap for future zero-copy reads
        if self._mm is not None:
            self._mm.close()
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        # Update in-memory index card and vector list
        self._index[key] = (offset, total_len)
        self._keys_list.append(key)
        self._vectors_list.append(vector.copy())

    def close(self) -> None:
        """Safely close open file handles and memory maps."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if not self._file.closed:
            self._file.close()
