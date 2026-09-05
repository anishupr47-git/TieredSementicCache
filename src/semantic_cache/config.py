"""
Tiered Semantic Cache - Configuration
====================================

What is this file?
------------------
This file holds all the settings (configuration) for our cache system in one safe,
unchangeable (immutable) package.

Simple Math & Logic for Everyone:
---------------------------------
* RAM Capacity (ram_capacity):
  - How many items fit in fast computer memory (RAM).
  - Think of it like the top of your office desk.
  - If your desk gets full, the oldest, least-used item automatically slides
    into the filing cabinet (L2 Disk).

* Similarity Threshold (similarity_threshold):
  - A dial from 0.0 (0%) to 1.0 (100%) that decides how close two meanings
    must be to count as a match.
  - 1.0 = Exactly identical meaning.
  - 0.70 (Default) = 70% match in meaning (e.g., "install python windows"
    matches "How do I install Python on Windows?").
  - 0.0 = Anything matches everything.

* Vector Dimension (vector_dim):
  - The number of coordinates (numbers) used to represent a sentence's meaning
    (e.g., 384 numbers).

* Time-To-Live (default_ttl):
  - How many seconds an answer stays valid before it expires and gets thrown away.

* Active Sweep & Interval (enable_active_sweep, sweep_interval_sec):
  - A quiet background helper that wakes up every few seconds (e.g., 30s)
    to throw away expired items so they never waste memory.

* Disk Path (disk_path):
  - The file on your hard drive where older items are saved.

* Port & Host (port, host):
  - The network address and door number where other apps talk to this cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CacheConfig:
    """Settings for the cache system. Once created, these values cannot be accidentally changed."""

    ram_capacity: int = 1000
    similarity_threshold: float = 0.70
    disk_path: Path = Path("cache.db")
    vector_dim: int = 384
    port: int = 6380
    host: str = "127.0.0.1"
    default_ttl: Optional[int] = None
    enable_active_sweep: bool = True
    sweep_interval_sec: float = 30.0
    requirepass: Optional[str] = None
    max_connections: int = 1000
    auto_compact_waste_ratio: float = 0.5
    enable_index_file: bool = False

    def __post_init__(self) -> None:
        """Validate all settings before starting the cache."""
        # 1. Check RAM capacity is a positive whole number (booleans not allowed)
        if type(self.ram_capacity) is not int or self.ram_capacity <= 0:
            raise ValueError(
                f"ram_capacity must be a positive integer greater than 0, got {self.ram_capacity}"
            )

        # 2. Check similarity threshold dial is between 0.0 (0%) and 1.0 (100%)
        if not isinstance(self.similarity_threshold, (int, float)) or isinstance(self.similarity_threshold, bool) or not (
            0.0 <= float(self.similarity_threshold) <= 1.0
        ):
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )

        # 3. Check vector size is a positive whole number
        if type(self.vector_dim) is not int or self.vector_dim <= 0:
            raise ValueError(
                f"vector_dim must be a positive integer greater than 0, got {self.vector_dim}"
            )

        # 4. Check network port is between 1 and 65535
        if type(self.port) is not int or not (1 <= self.port <= 65535):
            raise ValueError(
                f"port must be a valid network port between 1 and 65535, got {self.port}"
            )

        # 5. Check default_ttl if provided
        if self.default_ttl is not None and (
            type(self.default_ttl) is not int or self.default_ttl <= 0
        ):
            raise ValueError(f"default_ttl must be a positive integer, got {self.default_ttl}")

        # 6. Check sweep interval is positive
        if not isinstance(self.sweep_interval_sec, (int, float)) or self.sweep_interval_sec <= 0:
            raise ValueError(
                f"sweep_interval_sec must be positive, got {self.sweep_interval_sec}"
            )

        # 7. Sanitise and validate disk_path (prevents path traversal attacks)
        if isinstance(self.disk_path, str):
            object.__setattr__(self, "disk_path", Path(self.disk_path))
        elif not isinstance(self.disk_path, Path):
            raise TypeError(
                f"disk_path must be a string or Path, got {type(self.disk_path).__name__}"
            )
        resolved = self.disk_path.resolve()
        for part in resolved.parts:
            if part == "..":
                raise ValueError(
                    f"disk_path must not contain '..' segments, got {self.disk_path}"
                )
        object.__setattr__(self, "disk_path", resolved)

        # 8. Ensure threshold is stored as a float number
        if isinstance(self.similarity_threshold, int):
            object.__setattr__(self, "similarity_threshold", float(self.similarity_threshold))

        # 9. Check max_connections is a positive whole number
        if type(self.max_connections) is not int or self.max_connections <= 0:
            raise ValueError(
                f"max_connections must be a positive integer, got {self.max_connections}"
            )

        # 10. Check requirepass is a non-empty string if provided
        if self.requirepass is not None:
            if not isinstance(self.requirepass, str) or not self.requirepass.strip():
                raise ValueError("requirepass must be a non-empty string if set")

        # 11. Check auto_compact_waste_ratio is between 0.0 and 1.0
        if (
            not isinstance(self.auto_compact_waste_ratio, (int, float))
            or isinstance(self.auto_compact_waste_ratio, bool)
            or not (0.0 <= float(self.auto_compact_waste_ratio) <= 1.0)
        ):
            raise ValueError(
                f"auto_compact_waste_ratio must be a float between 0.0 and 1.0, got {self.auto_compact_waste_ratio}"
            )
        if isinstance(self.auto_compact_waste_ratio, int):
            object.__setattr__(self, "auto_compact_waste_ratio", float(self.auto_compact_waste_ratio))

        # 12. Check enable_index_file is a boolean
        if not isinstance(self.enable_index_file, bool):
            raise TypeError(
                f"enable_index_file must be a boolean, got {type(self.enable_index_file).__name__}"
            )

