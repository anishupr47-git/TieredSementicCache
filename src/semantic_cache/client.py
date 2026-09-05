"""
Tiered Semantic Cache - High-Performance Python Client SDK
==========================================================

What is this file?
------------------
This file provides `SemanticCacheClient`, a production-grade Python client
for communicating with the standalone Semantic Cache TCP server daemon.

Features:
- Speaks native RESP protocol over streaming TCP sockets.
- Automatic reconnect on transient socket drops.
- Optional in-process fallback cache if the remote TCP server is unreachable.
- Sub-millisecond latency with standard Redis commands.
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, Optional

from semantic_cache.protocol import RESPParser, RESPSerializer


class SemanticCacheClient:
    """Client for querying the Semantic Cache TCP server daemon over network."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6380,
        timeout: float = 3.0,
        fallback_cache: Any = None,
    ) -> None:
        """Initialize client connection configuration.

        Args:
            host: Target cache server hostname/IP.
            port: Target cache server TCP port.
            timeout: Socket timeout in seconds.
            fallback_cache: Optional in-process TieredSemanticCache to fallback to if server is unreachable.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.fallback = fallback_cache
        self._sock: Optional[socket.socket] = None
        self._file: Optional[Any] = None

    def _connect(self) -> None:
        """Establish or verify socket connection."""
        if self._sock is not None:
            return
        try:
            s = socket.create_connection((self.host, self.port), timeout=self.timeout)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self._sock = s
            self._file = s.makefile("rwb")
        except Exception:
            self._disconnect()
            raise

    def _disconnect(self) -> None:
        """Safely close active socket."""
        if self._file:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _send_command(self, *args: str) -> Any:
        """Send a RESP array command and read the response."""
        try:
            self._connect()
            assert self._file is not None

            # Encode as standard RESP Array
            items = [arg.encode("utf-8") for arg in args]
            payload = f"*{len(items)}\r\n".encode("utf-8")
            for it in items:
                payload += f"${len(it)}\r\n".encode("utf-8") + it + b"\r\n"

            self._file.write(payload)
            self._file.flush()

            # Read RESP response header
            line = self._file.readline()
            if not line:
                raise ConnectionResetError("Server closed connection")

            prefix = line[:1]
            content = line[1:].rstrip(b"\r\n")

            # Simple string (+)
            if prefix == b"+":
                return content.decode("utf-8")

            # Error (-)
            if prefix == b"-":
                raise RuntimeError(content.decode("utf-8"))

            # Integer (:)
            if prefix == b":":
                return int(content)

            # Bulk string ($)
            if prefix == b"$":
                length = int(content)
                if length == -1:
                    return None  # Cache Miss!
                data = self._file.read(length + 2)  # payload + \r\n
                return data[:-2].decode("utf-8")

            raise ValueError(f"Unexpected RESP response: {line!r}")

        except Exception:
            self._disconnect()
            raise

    def ping(self, message: Optional[str] = None) -> str:
        """Check server health."""
        try:
            if message:
                return str(self._send_command("PING", message))
            return str(self._send_command("PING"))
        except Exception:
            if self.fallback is not None:
                return message if message else "PONG"
            raise

    def get(self, query: str) -> Optional[str]:
        """Retrieve answer for query (exact or semantic fuzzy match)."""
        try:
            res = self._send_command("SEMANTIC.GET", query)
            return res
        except Exception:
            if self.fallback is not None:
                hit = self.fallback.get(query)
                return hit.value if hit else None
            raise

    def set(
        self,
        query: str,
        answer: str,
        ttl: Optional[int] = None,
    ) -> bool:
        """Store answer with optional TTL in seconds."""
        try:
            if ttl is not None and ttl > 0:
                res = self._send_command("SEMANTIC.SETEX", query, str(ttl), answer)
            else:
                res = self._send_command("SEMANTIC.SET", query, answer)
            return res == "OK"
        except Exception:
            if self.fallback is not None:
                self.fallback.set(query, answer, ttl=ttl)
                return True
            raise

    def delete(self, query: str) -> bool:
        """Delete query from cache."""
        try:
            res = self._send_command("DEL", query)
            return bool(res == 1)
        except Exception:
            if self.fallback is not None:
                return self.fallback.delete(query)
            raise

    def expire(self, query: str, seconds: float) -> bool:
        """Set or update TTL on key."""
        try:
            res = self._send_command("EXPIRE", query, str(seconds))
            return bool(res == 1)
        except Exception:
            if self.fallback is not None:
                return self.fallback.expire(query, seconds)
            raise

    def ttl(self, query: str) -> int:
        """Return remaining TTL seconds (-2 if missing, -1 if no TTL, >=0 remaining)."""
        try:
            return int(self._send_command("TTL", query))
        except Exception:
            if self.fallback is not None:
                return self.fallback.ttl(query)
            raise

    def dbsize(self) -> int:
        """Return total count of cached items."""
        try:
            return int(self._send_command("DBSIZE"))
        except Exception:
            if self.fallback is not None:
                return len(self.fallback)
            raise

    def invalidate_tag(self, tag: str) -> int:
        """Invalidate all items belonging to a tag."""
        try:
            return int(self._send_command("TAG.INVALIDATE", tag))
        except Exception:
            if self.fallback is not None:
                return self.fallback.invalidate_tag(tag)
            raise

    def stats(self) -> Dict[str, Any]:
        """Fetch server operational statistics as a dictionary."""
        try:
            raw_json = self._send_command("STATS")
            return json.loads(raw_json)
        except Exception:
            if self.fallback is not None:
                return self.fallback.stats()
            raise

    def compact(self) -> int:
        """Trigger disk log compaction and return reclaimed bytes."""
        try:
            return int(self._send_command("COMPACT"))
        except Exception:
            if self.fallback is not None:
                return self.fallback.compact()
            raise

    def flushdb(self) -> bool:
        """Clear cache completely."""
        try:
            res = self._send_command("FLUSHDB")
            return res == "OK"
        except Exception:
            if self.fallback is not None:
                self.fallback.clear()
                return True
            raise

    def close(self) -> None:
        """Close connection."""
        self._disconnect()

    def __enter__(self) -> SemanticCacheClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
