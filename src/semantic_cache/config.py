"""
Tiered Semantic Cache - Configuration
====================================

What is this file?
------------------
This file holds all the settings (configuration) for our cache system in one safe,
unchangeable (immutable) package.

Simple Math & Logic in Points:
------------------------------
* RAM Capacity (ram_capacity):
  - The maximum number of items kept in fast computer memory (RAM).
  - Rule: If the cache gets full, the oldest, least-used item is moved to the hard drive.

* Similarity Threshold (similarity_threshold, or tau):
  - A dial from 0.0 to 1.0 that decides how close two meanings must be to count as a match.
  - 1.0 = Sentences must have the exact same direction/meaning.
  - 0.70 (Default) = 70% close in meaning (catches "install python windows" vs "How do I install Python on Windows?").
  - 0.0 = Anything matches everything.

* Vector Dimension (vector_dim):
  - The list size of numbers used to represent a sentence's meaning (e.g., 384 numbers).

* Disk Path (disk_path):
  - The file on your hard drive where older items are saved so we never run out of RAM.

* Port & Host (port, host):
  - The network door number and address where other apps connect to talk to this cache.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CacheConfig:
    """Settings for the cache system. Once created, these values cannot be accidentally changed.

    Attributes:
        ram_capacity: Maximum items allowed in RAM (must be > 0).
        similarity_threshold: Closeness dial between 0.0 and 1.0 (default 0.85).
        disk_path: File location on hard drive to store overflow data.
        vector_dim: How many numbers describe each sentence (must be > 0).
        port: Network door number to listen on (1 to 65535, default 6380).
        host: Network address to listen on (default '127.0.0.1' = this computer only).
    """

    ram_capacity: int = 1000
    similarity_threshold: float = 0.70
    disk_path: Path = Path("cache.db")
    vector_dim: int = 384
    port: int = 6380
    host: str = "127.0.0.1"
    default_ttl: Optional[int] = None
    enable_active_sweep: bool = True
    sweep_interval_sec: float = 30.0

    def __post_init__(self) -> None:
        """Simple checks to catch any mistake before starting the cache."""
        # 1. Check RAM capacity is positive (and not boolean)
        if type(self.ram_capacity) is not int or self.ram_capacity <= 0:
            raise ValueError(
                f"ram_capacity must be a positive integer greater than 0, got {self.ram_capacity}"
            )

        # 2. Check similarity dial is between 0.0 (0%) and 1.0 (100%)
        if not isinstance(self.similarity_threshold, (int, float)) or isinstance(self.similarity_threshold, bool) or not (
            0.0 <= float(self.similarity_threshold) <= 1.0
        ):
            raise ValueError(
                f"similarity_threshold must be between 0.0 and 1.0, got {self.similarity_threshold}"
            )

        # 3. Check vector size is positive (and not boolean)
        if type(self.vector_dim) is not int or self.vector_dim <= 0:
            raise ValueError(
                f"vector_dim must be a positive integer greater than 0, got {self.vector_dim}"
            )

        # 4. Check network port is valid (and not boolean)
        if type(self.port) is not int or not (1 <= self.port <= 65535):
            raise ValueError(
                f"port must be a valid network port between 1 and 65535, got {self.port}"
            )

        # 5. Check default_ttl if provided
        if self.default_ttl is not None and (
            type(self.default_ttl) is not int or self.default_ttl <= 0
        ):
            raise ValueError(f"default_ttl must be a positive integer, got {self.default_ttl}")

        # 6. Check sweep interval
        if not isinstance(self.sweep_interval_sec, (int, float)) or self.sweep_interval_sec <= 0:
            raise ValueError(
                f"sweep_interval_sec must be positive, got {self.sweep_interval_sec}"
            )

        # 5. Convert text file path to a proper Path object
        if isinstance(self.disk_path, str):
            object.__setattr__(self, "disk_path", Path(self.disk_path))
        elif not isinstance(self.disk_path, Path):
            raise TypeError(
                f"disk_path must be a string or Path, got {type(self.disk_path).__name__}"
            )

        # 6. Ensure threshold is stored as a float number
        if isinstance(self.similarity_threshold, int):
            object.__setattr__(self, "similarity_threshold", float(self.similarity_threshold))
