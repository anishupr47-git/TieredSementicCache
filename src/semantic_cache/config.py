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

import os
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
    ssl_certfile: Optional[Path | str] = None
    ssl_keyfile: Optional[Path | str] = None
    ssl_ca_certs: Optional[Path | str] = None

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

        # 10. Check requirepass: fallback to env var if not set, and validate non-empty string
        if self.requirepass is None:
            env_pass = os.environ.get("CACHE_REQUIREPASS") or os.environ.get("TIERED_CACHE_PASSWORD")
            if env_pass and env_pass.strip():
                object.__setattr__(self, "requirepass", env_pass.strip())

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

        # 13. Validate TLS/SSL options if provided
        if self.ssl_certfile is not None:
            cert_p = Path(self.ssl_certfile).resolve()
            if not cert_p.is_file():
                raise FileNotFoundError(f"ssl_certfile does not exist: {self.ssl_certfile}")
            object.__setattr__(self, "ssl_certfile", cert_p)

        if self.ssl_keyfile is not None:
            if self.ssl_certfile is None:
                raise ValueError("ssl_certfile must be specified when ssl_keyfile is set")
            key_p = Path(self.ssl_keyfile).resolve()
            if not key_p.is_file():
                raise FileNotFoundError(f"ssl_keyfile does not exist: {self.ssl_keyfile}")
            object.__setattr__(self, "ssl_keyfile", key_p)

        if self.ssl_ca_certs is not None:
            ca_p = Path(self.ssl_ca_certs).resolve()
            if not ca_p.is_file():
                raise FileNotFoundError(f"ssl_ca_certs does not exist: {self.ssl_ca_certs}")
            object.__setattr__(self, "ssl_ca_certs", ca_p)

    def __repr__(self) -> str:
        """Safe representation with secret masking for sensitive credentials."""
        masked_pass = "'***'" if self.requirepass is not None else "None"
        return (
            f"CacheConfig("
            f"ram_capacity={self.ram_capacity}, "
            f"similarity_threshold={self.similarity_threshold}, "
            f"disk_path={self.disk_path!r}, "
            f"vector_dim={self.vector_dim}, "
            f"port={self.port}, "
            f"host={self.host!r}, "
            f"default_ttl={self.default_ttl}, "
            f"enable_active_sweep={self.enable_active_sweep}, "
            f"sweep_interval_sec={self.sweep_interval_sec}, "
            f"requirepass={masked_pass}, "
            f"max_connections={self.max_connections}, "
            f"auto_compact_waste_ratio={self.auto_compact_waste_ratio}, "
            f"enable_index_file={self.enable_index_file}, "
            f"ssl_certfile={self.ssl_certfile!r}"
            f")"
        )


