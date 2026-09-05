"""
Tiered Semantic Cache - Python Client SDK (The "Remote Control")
================================================================

What is this file?
------------------
This file provides `SemanticCacheClient`, a Python remote control for talking
to the Semantic Cache server over the network.

Key Features for Everyone:
--------------------------
1. Standard Redis Wire Language (RESP):
   - Speaks standard Redis protocol over fast TCP connections.
   - Compatible with local servers or remote VPS cloud instances.

2. Crash-Proof In-Process Fallback:
   - If your server or VPS ever goes offline, the client can automatically
     fall back to a local in-memory cache on your computer.
   - Your website or chatbot will NEVER crash!

3. Microsecond Speed:
   - Reuses open socket connections and uses zero-delay TCP buffering (TCP_NODELAY).
"""

from __future__ import annotations

import json
import socket
import time
from typing import Any, Dict, Optional

from semantic_cache.protocol import RESPParser, RESPSerializer


class SemanticCacheClient:
    """Client for connecting to the Semantic Cache server over the network."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 6380,
        timeout: float = 3.0,
        fallback_cache: Any = None,
    ) -> None:
        """Set up client connection settings.

        Args:
            host: Server address (default '127.0.0.1' = this computer).
            port: Server door number (default 6380).
            timeout: How many seconds to wait for a response before timing out.
            fallback_cache: Local cache to use automatically if the server is offline.
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.fallback = fallback_cache
        self._sock: Optional[socket.socket] = None
        self._file: Optional[Any] = None

    def _connect(self) -> None:
        """Open or check the network connection."""
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
        """Safely close the connection."""
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
        """Send a command to the server and read back the answer."""
        try:
            self._connect()
            assert self._file is not None

            # Package command into standard Redis bytes
            items = [arg.encode("utf-8") for arg in args]
            payload = f"*{len(items)}\r\n".encode("utf-8")
            for it in items:
                payload += f"${len(it)}\r\n".encode("utf-8") + it + b"\r\n"

            self._file.write(payload)
            self._file.flush()

            # Read the response
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
                data = self._file.read(length + 2)
                return data[:-2].decode("utf-8")

            raise ValueError(f"Unexpected response from server: {line!r}")

        except Exception:
            self._disconnect()
            raise

    def ping(self, message: Optional[str] = None) -> str:
        """Check if the server is alive and responding."""
        try:
            if message:
                return str(self._send_command("PING", message))
            return str(self._send_command("PING"))
        except Exception:
            if self.fallback is not None:
                return message if message else "PONG"
            raise

    def get(self, query: str) -> Optional[str]:
        """Ask for a cached answer (checks both exact text and similar meanings)."""
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
        """Save a question and answer with an optional countdown timer (TTL)."""
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
        """Delete an answer from the cache."""
        try:
            res = self._send_command("DEL", query)
            return bool(res == 1)
        except Exception:
            if self.fallback is not None:
                return self.fallback.delete(query)
            raise

    def expire(self, query: str, seconds: float) -> bool:
        """Set or update an expiration countdown timer on a saved answer."""
        try:
            res = self._send_command("EXPIRE", query, str(seconds))
            return bool(res == 1)
        except Exception:
            if self.fallback is not None:
                return self.fallback.expire(query, seconds)
            raise

    def ttl(self, query: str) -> int:
        """Check remaining seconds before an answer expires (-2 if missing, -1 if no timer)."""
        try:
            return int(self._send_command("TTL", query))
        except Exception:
            if self.fallback is not None:
                return self.fallback.ttl(query)
            raise

    def dbsize(self) -> int:
        """Check total number of answers saved in the cache."""
        try:
            return int(self._send_command("DBSIZE"))
        except Exception:
            if self.fallback is not None:
                return len(self.fallback)
            raise

    def invalidate_tag(self, tag: str) -> int:
        """Delete all answers labeled with a given tag."""
        try:
            return int(self._send_command("TAG.INVALIDATE", tag))
        except Exception:
            if self.fallback is not None:
                return self.fallback.invalidate_tag(tag)
            raise

    def stats(self) -> Dict[str, Any]:
        """Fetch server operational statistics (hits, misses, item counts)."""
        try:
            raw_json = self._send_command("STATS")
            return json.loads(raw_json)
        except Exception:
            if self.fallback is not None:
                return self.fallback.stats()
            raise

    def compact(self) -> int:
        """Clean up disk file on server to reclaim wasted storage space."""
        try:
            return int(self._send_command("COMPACT"))
        except Exception:
            if self.fallback is not None:
                return self.fallback.compact()
            raise

    def flushdb(self) -> bool:
        """Wipe the entire cache clean."""
        try:
            res = self._send_command("FLUSHDB")
            return res == "OK"
        except Exception:
            if self.fallback is not None:
                self.fallback.clear()
                return True
            raise

    def close(self) -> None:
        """Close the network connection."""
        self._disconnect()

    def __enter__(self) -> SemanticCacheClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
