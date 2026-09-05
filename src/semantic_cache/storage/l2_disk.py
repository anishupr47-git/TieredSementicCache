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

Binary File Format (Append-Only Log with CRC32 Integrity):
-----------------------------------------------------------
Each saved item is written to the end of the file as compact binary bytes:
┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┬──────────────────┐
│ key_len  │ val_len  │ vec_dim  │ expires  │ tag_len  │  crc32   │     payload      │
│  (u32)   │  (u32)   │  (u32)   │  (u64)   │  (u32)   │  (u32)   │  (text + vector) │
│ 4 bytes  │ 4 bytes  │ 4 bytes  │ 8 bytes  │ 4 bytes  │ 4 bytes  │                  │
└──────────┴──────────┴──────────┴──────────┴──────────┴──────────┴──────────────────┘

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

import heapq
import logging
import mmap
import os
from pathlib import Path
import struct
import time
from typing import Optional, Sequence, Tuple
import zlib
import numpy as np

logger = logging.getLogger("semantic_cache.l2_disk")

from semantic_cache.storage.l1_ram import CacheRecord

# Binary Record Header Format (v2 with CRC32 integrity):
# klen(u32), vlen(u32), dim(u32), expires_at_ms(u64), tag_len(u32), crc32(u32) = 28 bytes
HEADER_FORMAT = "<IIIQII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


class L2DiskCache:
    """Zero-copy memory-mapped append-only disk storage with TTL, tags, and compaction."""

    def __init__(self, file_path: Path, dim: int = 384, enable_index_file: bool = False) -> None:
        """Set up filing cabinet at the given file path."""
        self.file_path = Path(file_path)
        self.dim = dim
        self.enable_index_file = enable_index_file
        self.index_path = self.file_path.with_name(self.file_path.name + ".idx")

        # Make sure storage folder exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)

        # Quick index card map: key -> (byte_offset, total_record_length)
        self._index: dict[str, Tuple[int, int]] = {}
        self._key_to_slot: dict[str, int] = {}
        self._key_expires: dict[str, float] = {}

        # Contiguous table of arrows for fast vector matching
        self._keys_list: list[str] = []
        self._matrix = np.empty((128, dim), dtype=np.float32)
        self._count: int = 0

        # TC-1: Expiry min-heap for O(K) sweep instead of O(N) full scan
        self._expiry_heap: list[Tuple[float, str]] = []

        # Open file handle and memory map
        self._file = open(self.file_path, "a+b")
        self._mm: Optional[mmap.mmap] = None

        # TC-3: Load from fast index file or scan binary log
        loaded_from_idx = False
        if self.enable_index_file:
            loaded_from_idx = self._load_index()

        if not loaded_from_idx:
            self._build_index()
            if self.enable_index_file and self._count > 0:
                self.save_index()

        logger.info(
            "L2DiskCache loaded: %d records from %s (from_index=%s)",
            self._count, self.file_path, loaded_from_idx,
        )

    def __len__(self) -> int:
        """Count total unique answers in the filing cabinet."""
        return self._count

    def __contains__(self, key: str) -> bool:
        """Check if an answer is in the cabinet (removes if expired)."""
        if key not in self._index:
            return False
        rec = self.get_exact(key)
        return rec is not None

    def save_index(self) -> bool:
        """TC-3: Save the in-memory index card map and matrix to an index file for fast startup."""
        try:
            self.index_path.parent.mkdir(parents=True, exist_ok=True)
            temp_idx = self.index_path.with_suffix(".tmp")
            self._file.seek(0, 2)
            db_size = self._file.tell()
            try:
                mtime_ns = os.stat(self.file_path).st_mtime_ns
            except OSError:
                mtime_ns = 0

            magic = b"TSCIDX01"
            header = struct.pack("<IIQQQ", 1, self.dim, self._count, db_size, mtime_ns)

            entries_bytes = bytearray()
            for k in self._keys_list[: self._count]:
                off, rlen = self._index[k]
                exp_sec = self._key_expires.get(k)
                exp_ms = int(exp_sec * 1000.0) if exp_sec is not None and exp_sec > 0 else 0
                kbytes = k.encode("utf-8")
                entries_bytes.extend(struct.pack("<HIIQ", len(kbytes), off, rlen, exp_ms))
                entries_bytes.extend(kbytes)

            matrix_bytes = self._matrix[: self._count].tobytes()
            payload = magic + header + bytes(entries_bytes) + matrix_bytes
            crc = zlib.crc32(payload) & 0xFFFFFFFF
            full_data = payload + struct.pack("<I", crc)

            with open(temp_idx, "wb") as f:
                f.write(full_data)
                f.flush()
                os.fsync(f.fileno())

            os.replace(temp_idx, self.index_path)
            logger.debug(
                "Saved index file (%d records, %d bytes) to %s",
                self._count, len(full_data), self.index_path,
            )
            return True
        except Exception as e:
            logger.warning("Failed to save index file %s: %s", self.index_path, e, exc_info=True)
            return False

    def _load_index(self) -> bool:
        """TC-3: Attempt to restore cache state from index file (.idx) for fast O(1) startup."""
        if not self.index_path.exists():
            return False

        try:
            with open(self.index_path, "rb") as f:
                data = f.read()

            if len(data) < 44:  # 8 (magic) + 32 (header) + 4 (crc)
                logger.warning("Index file %s is too small, falling back to full scan", self.index_path)
                return False

            # Verify CRC32 checksum
            stored_crc = struct.unpack_from("<I", data, len(data) - 4)[0]
            computed_crc = zlib.crc32(data[:-4]) & 0xFFFFFFFF
            if stored_crc != computed_crc:
                logger.warning(
                    "Index file CRC32 mismatch (%08x != %08x), falling back to full scan",
                    stored_crc, computed_crc,
                )
                return False

            if data[:8] != b"TSCIDX01":
                logger.warning("Index file magic bytes invalid, falling back to full scan")
                return False

            version, dim, count, saved_db_size, saved_mtime_ns = struct.unpack_from("<IIQQQ", data, 8)
            if version != 1 or dim != self.dim:
                logger.warning("Index version (%d) or dim (%d != %d) mismatch", version, dim, self.dim)
                return False

            self._file.seek(0, 2)
            current_db_size = self._file.tell()
            if current_db_size < saved_db_size:
                logger.warning(
                    "Data file size (%d) is smaller than saved index (%d), falling back to full scan",
                    current_db_size, saved_db_size,
                )
                return False

            # Unpack entries
            pos = 40
            index_map: dict[str, Tuple[int, int]] = {}
            keys_list: list[str] = []
            key_to_slot: dict[str, int] = {}
            key_expires: dict[str, float] = {}
            expiry_heap: list[Tuple[float, str]] = []

            for slot in range(count):
                klen, off, rlen, exp_ms = struct.unpack_from("<HIIQ", data, pos)
                pos += 18
                key = data[pos : pos + klen].decode("utf-8")
                pos += klen

                index_map[key] = (off, rlen)
                keys_list.append(key)
                key_to_slot[key] = slot
                if exp_ms > 0:
                    exp_sec = exp_ms / 1000.0
                    key_expires[key] = exp_sec
                    heapq.heappush(expiry_heap, (exp_sec, key))

            matrix_bytes_len = count * dim * 4
            if pos + matrix_bytes_len != len(data) - 4:
                logger.warning("Index file length mismatch, falling back to full scan")
                return False

            raw_matrix = np.frombuffer(
                data[pos : pos + matrix_bytes_len], dtype=np.float32
            ).reshape((count, dim))

            alloc_cap = max(128, max(count * 2, 128))
            self._matrix = np.empty((alloc_cap, self.dim), dtype=np.float32)
            if count > 0:
                self._matrix[:count] = raw_matrix

            self._count = count
            self._index = index_map
            self._keys_list = keys_list
            self._key_to_slot = key_to_slot
            self._key_expires = key_expires
            self._expiry_heap = expiry_heap

            # If new records were appended since index was saved, scan the tail
            if current_db_size > saved_db_size:
                logger.debug(
                    "Fast-forwarding index from offset %d to %d (%d bytes)",
                    saved_db_size, current_db_size, current_db_size - saved_db_size,
                )
                self._scan_tail(saved_db_size, current_db_size)
                self.save_index()

            # Sweep any records that expired in the meantime
            self.sweep_expired()
            return True
        except Exception as e:
            logger.warning("Failed to load index file %s: %s, falling back to full scan", self.index_path, e, exc_info=True)
            self._index.clear()
            self._keys_list.clear()
            self._key_to_slot.clear()
            self._key_expires.clear()
            self._expiry_heap.clear()
            self._count = 0
            return False

    def _scan_tail(self, start_offset: int, end_offset: int) -> None:
        """Scan newly appended records between start_offset and end_offset."""
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        self._mm = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)

        offset = start_offset
        while offset + HEADER_SIZE <= end_offset:
            klen, vlen, dim, exp_ms, tag_len, stored_crc = struct.unpack_from(HEADER_FORMAT, self._mm, offset)
            record_len = HEADER_SIZE + klen + vlen + tag_len + (dim * 4)
            if offset + record_len > end_offset:
                logger.warning("Truncated tail record at offset %d", offset)
                break

            payload_start = offset + HEADER_SIZE
            payload_bytes = bytes(self._mm[payload_start : payload_start + klen + vlen + tag_len + (dim * 4)])
            computed_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
            if stored_crc != 0 and computed_crc != stored_crc:
                logger.warning("CRC32 mismatch in tail record at offset %d", offset)
                offset += record_len
                continue

            expires_at_sec = (exp_ms / 1000.0) if exp_ms > 0 else None
            key = self._mm[payload_start : payload_start + klen].decode("utf-8")
            vec_offset = payload_start + klen + vlen + tag_len
            vec = np.frombuffer(self._mm, dtype=np.float32, count=dim, offset=vec_offset).copy()

            if expires_at_sec is not None and time.time() >= expires_at_sec:
                if key in self._index:
                    self.remove(key)
                offset += record_len
                continue

            if key in self._key_to_slot:
                slot = self._key_to_slot[key]
                self._matrix[slot] = vec
                self._index[key] = (offset, record_len)
            else:
                slot = self._count
                if slot >= len(self._matrix):
                    new_cap = max(128, len(self._matrix) * 2)
                    new_matrix = np.empty((new_cap, self.dim), dtype=np.float32)
                    new_matrix[:slot] = self._matrix[:slot]
                    self._matrix = new_matrix
                self._matrix[slot] = vec
                self._keys_list.append(key)
                self._key_to_slot[key] = slot
                self._index[key] = (offset, record_len)
                self._count += 1

            if expires_at_sec is not None:
                self._key_expires[key] = expires_at_sec
                heapq.heappush(self._expiry_heap, (expires_at_sec, key))
            else:
                self._key_expires.pop(key, None)

            offset += record_len

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
        deduped: dict[str, Tuple[int, int, np.ndarray, Optional[float]]] = {}

        while offset + HEADER_SIZE <= size:
            klen, vlen, dim, exp_ms, tag_len, stored_crc = struct.unpack_from(HEADER_FORMAT, self._mm, offset)
            record_len = HEADER_SIZE + klen + vlen + tag_len + (dim * 4)

            if offset + record_len > size:
                logger.warning("Truncated record at offset %d (expected %d bytes, only %d remain)", offset, record_len, size - offset)
                break  # Incomplete record at the end of the file

            # CRC32 integrity check: verify payload hasn't been corrupted
            payload_start = offset + HEADER_SIZE
            payload_bytes = bytes(self._mm[payload_start : payload_start + klen + vlen + tag_len + (dim * 4)])
            computed_crc = zlib.crc32(payload_bytes) & 0xFFFFFFFF
            if stored_crc != 0 and computed_crc != stored_crc:
                logger.warning(
                    "CRC32 mismatch at offset %d (stored=%08x computed=%08x), skipping corrupt record",
                    offset, stored_crc, computed_crc,
                )
                offset += record_len
                continue

            # Skip expired entries
            expires_at_sec = (exp_ms / 1000.0) if exp_ms > 0 else None
            if expires_at_sec is not None and time.time() >= expires_at_sec:
                offset += record_len
                continue

            # Read key text
            key_bytes = self._mm[payload_start : payload_start + klen]
            key = key_bytes.decode("utf-8")

            # Read arrow vector
            vec_offset = payload_start + klen + vlen + tag_len
            vec = np.frombuffer(self._mm, dtype=np.float32, count=dim, offset=vec_offset).copy()

            # Latest version always wins
            deduped[key] = (offset, record_len, vec, expires_at_sec)
            offset += record_len

        # Fill table with clean, unique items
        self._count = len(deduped)
        if self._count > 0:
            alloc_cap = max(128, self._count * 2)
            self._matrix = np.empty((alloc_cap, self.dim), dtype=np.float32)
            for slot, (k, (off, rlen, vec, exp_sec)) in enumerate(deduped.items()):
                self._index[k] = (off, rlen)
                self._keys_list.append(k)
                self._key_to_slot[k] = slot
                self._matrix[slot] = vec
                # TC-1: Populate expiry heap for items with TTL
                if exp_sec is not None:
                    self._key_expires[k] = exp_sec
                    heapq.heappush(self._expiry_heap, (exp_sec, k))

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
        klen, vlen, dim, exp_ms, tag_len, _crc = struct.unpack_from(HEADER_FORMAT, self._mm, offset)

        expires_at = (exp_ms / 1000.0) if exp_ms > 0 else None
        if expires_at is not None and time.time() >= expires_at:
            self.remove(key)
            return None

        # Read value string, tags, and vector directly from the memory map
        payload_start = offset + HEADER_SIZE
        val_start = payload_start + klen
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
        key_prefix: Optional[str] = None,
    ) -> Optional[Tuple[CacheRecord, float]]:
        """Search disk entries for meaning similarity in 1 instant calculation.

        Args:
            key_prefix: If set, only match keys that start with this prefix.
                        Used by namespaced caches to prevent cross-tenant leaks.
        """
        if self._count == 0:
            return None

        # Hardware-accelerated dot product across all active arrows
        active_matrix = self._matrix[: self._count]
        scores = np.dot(active_matrix, query_vector)

        # SEC-4: Mask out keys that don't belong to this namespace
        if key_prefix is not None:
            for i, k in enumerate(self._keys_list[: self._count]):
                if not k.startswith(key_prefix):
                    scores[i] = -2.0  # Below any valid threshold

        # O(N) single-pass: find the best score first
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score < threshold:
            return None  # Nothing close enough

        best_key = self._keys_list[best_idx]
        record = self.get_exact(best_key)
        if record is not None:
            return record, best_score

        # Best was expired — fall back to sorted scan for remaining candidates
        candidate_indices = np.argsort(-scores)
        for idx in candidate_indices:
            score = float(scores[idx])
            if score < threshold:
                break

            cand_key = self._keys_list[idx]
            cand_record = self.get_exact(cand_key)
            if cand_record is not None:
                return cand_record, score

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
        val_bytes = value.encode("utf-8") if isinstance(value, str) else bytes(value)
        tag_bytes = ",".join(tags).encode("utf-8") if tags else b""
        vec_arr = np.asarray(vector, dtype=np.float32)
        vec_bytes = vec_arr.tobytes()

        klen = len(key_bytes)
        vlen = len(val_bytes)
        tag_len = len(tag_bytes)
        dim = len(vec_arr)
        exp_ms = int(expires_at * 1000.0) if expires_at is not None and expires_at > 0 else 0

        # CRC32 integrity checksum covers all payload bytes
        payload_data = key_bytes + val_bytes + tag_bytes + vec_bytes
        crc = zlib.crc32(payload_data) & 0xFFFFFFFF

        header = struct.pack(HEADER_FORMAT, klen, vlen, dim, exp_ms, tag_len, crc)
        payload = header + payload_data
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

        # LAT-2: mmap is NOT reopened here. It will be lazily reopened
        # by get_exact() on the next read. This saves ~200-500µs per write
        # on Windows by avoiding the kernel mmap create/destroy cycle.

        # TC-1: Track expiry in the min-heap for fast sweep
        if expires_at is not None and expires_at > 0:
            self._key_expires[key] = expires_at
            heapq.heappush(self._expiry_heap, (expires_at, key))
        else:
            self._key_expires.pop(key, None)

        # Update index card and vector slot
        if key in self._key_to_slot:
            slot = self._key_to_slot[key]
            self._matrix[slot] = vec_arr
            self._index[key] = (offset, total_len)
        else:
            slot = self._count
            if slot >= len(self._matrix):
                new_cap = max(128, len(self._matrix) * 2)
                new_matrix = np.empty((new_cap, self.dim), dtype=np.float32)
                new_matrix[:slot] = self._matrix[:slot]
                self._matrix = new_matrix

            self._matrix[slot] = vec_arr
            self._keys_list.append(key)
            self._key_to_slot[key] = slot
            self._index[key] = (offset, total_len)
            self._count += 1

    def sweep_expired(self) -> int:
        """Toss out expired answers using the expiry heap (O(K) where K = expired count).

        Instead of scanning all N keys, we only pop items from the min-heap
        whose expiration time has passed. Much faster when N is large.
        """
        now = time.time()
        swept = 0
        while self._expiry_heap:
            earliest_exp, earliest_key = self._expiry_heap[0]
            if earliest_exp > now:
                break  # Everything left in the heap hasn't expired yet
            heapq.heappop(self._expiry_heap)
            # Key may have been updated or deleted since it was added to the heap
            if earliest_key in self._index:
                rec = self.get_exact(earliest_key)
                if rec is None and earliest_key not in self._index:
                    swept += 1
        return swept

    def remove(self, key: str) -> bool:
        """Remove an item from disk index and vector table using Card-Deck Swap."""
        if key not in self._index:
            return False

        del self._index[key]
        self._key_expires.pop(key, None)
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

    def waste_stats(self) -> Tuple[int, int, float]:
        """PR-6: Calculate (file_size, wasted_bytes, waste_ratio) to monitor TTL disk bloat.

        Returns:
            total_size: Current disk file size in bytes.
            wasted: Estimated wasted bytes from deleted or overwritten records.
            ratio: Fraction of file size that is dead space (0.0 to 1.0).
        """
        active_bytes = sum(rlen for _, rlen in self._index.values())
        try:
            self._file.seek(0, 2)
            total_size = self._file.tell()
        except (ValueError, OSError):
            total_size = 0
        wasted = max(0, total_size - active_bytes)
        ratio = (wasted / total_size) if total_size > 0 else 0.0
        return total_size, wasted, ratio

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
        logger.info(
            "Compaction finished: %d bytes reclaimed (%.1f%% waste)",
            max(0, reclaimed),
            (reclaimed / max(orig_size, 1)) * 100,
        )

        # TC-3: Keep index file in sync with compacted file
        if self.enable_index_file:
            self.save_index()

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
        self._key_expires.clear()
        self._matrix = np.empty((128, self.dim), dtype=np.float32)
        self._count = 0

        if self.enable_index_file and self.index_path.exists():
            try:
                self.index_path.unlink()
            except OSError:
                pass

    def close(self) -> None:
        """Safely close open file handles and memory maps."""
        # TC-3: Persist fast index on clean shutdown
        if self.enable_index_file and self._count > 0:
            try:
                self.save_index()
            except Exception as e:
                logger.warning("Failed to save index file on close: %s", e)

        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if not self._file.closed:
            self._file.close()
