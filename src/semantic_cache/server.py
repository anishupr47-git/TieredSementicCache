"""
Tiered Semantic Cache - AsyncIO TCP Server Daemon (The "Drive-Thru Window")
==========================================================================

What is this file?
------------------
This file runs our 24/7 background server.
Think of it like a 24/7 drive-thru window:
- Clients (Python apps, Node.js web servers, Go services, or redis-cli)
  connect over a standard network door (Port 6380).
- Uses Python's non-blocking AsyncIO event loop:
  A single computer thread can handle 10,000+ customer connections at the
  same time without breaking a sweat or wasting memory!

Supported Commands in Plain English:
------------------------------------
1. PING [message]
   - Health check. Returns "+PONG" or echoes your text.
2. SEMANTIC.SET <key> <value>
   - Turns the key into numbers (arrow), saves it in L1 RAM (and L2 Disk if full),
     and returns "+OK".
3. SEMANTIC.GET <key>
   - Searches for an exact match OR a meaning match (score >= threshold).
   - If found: returns the cached answer as a Bulk String ($len\\r\\nvalue\\r\\n).
   - If missed: returns standard Redis Null ($-1\\r\\n).
4. STATS
   - Returns cache health, item counts, hits, misses, and evictions.
5. QUIT
   - Politely closes the connection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
from typing import Optional

from semantic_cache.config import CacheConfig
from semantic_cache.embedder import BaseEmbedder, DenseHashEmbedder
from semantic_cache.protocol import RESPParser, RESPSerializer
from semantic_cache.storage.manager import StorageManager


class SemanticCacheServer:
    """Non-blocking TCP daemon coordinating RESP wire requests with the two-tier cache."""

    def __init__(
        self,
        config: Optional[CacheConfig] = None,
        embedder: Optional[BaseEmbedder] = None,
    ) -> None:
        """Initialize server with configuration, storage engine, and vector embedder."""
        self.config = config or CacheConfig()
        self.embedder = embedder or DenseHashEmbedder(dim=self.config.vector_dim)
        self.storage = StorageManager(self.config)
        self._server: Optional[asyncio.Server] = None
        self._running: bool = False

    async def start(self) -> None:
        """Start listening for incoming client connections."""
        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.config.host,
            port=self.config.port,
        )
        self._running = True

    async def stop(self) -> None:
        """Gracefully shut down the server and close storage files."""
        self._running = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self.storage.close()

    async def serve_forever(self) -> None:
        """Start the server and run until cancelled."""
        await self.start()
        print(f"[*] Semantic Cache Server running on {self.config.host}:{self.config.port}")
        print(f"[*] RAM Capacity: {self.config.ram_capacity} | Threshold: {self.config.similarity_threshold}")
        try:
            if self._server is not None:
                await self._server.serve_forever()
        except asyncio.CancelledError:
            pass
        finally:
            await self.stop()

    async def _handle_client(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle incoming commands from a single client connection."""
        try:
            while self._running:
                args = await RESPParser.read_command(reader)
                if args is None:
                    break  # Client closed connection or malformed payload

                response = self._execute_command(args)
                writer.write(response)
                await writer.drain()

                # If client requested QUIT, close connection
                if args and args[0].upper() == "QUIT":
                    break
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError, ValueError):
            pass
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except Exception:
                pass

    def _execute_command(self, args: list[str]) -> bytes:
        """Route and execute a parsed command in microseconds."""
        if not args:
            return RESPSerializer.error("empty command")

        cmd = args[0].upper()

        # 1. Health check: PING [message]
        if cmd == "PING":
            msg = args[1] if len(args) > 1 else None
            return RESPSerializer.pong(msg)

        # 2. Store: SEMANTIC.SET <key> <value> (also supports standard SET)
        if cmd in ("SEMANTIC.SET", "SET"):
            if len(args) < 3:
                return RESPSerializer.error(f"wrong number of arguments for '{cmd}'")
            key = args[1]
            val = args[2]

            vector = self.embedder.embed(key)
            self.storage.set(key, val, vector)
            return RESPSerializer.ok()

        # 3. Store with TTL: SETEX <key> <seconds> <value>
        if cmd in ("SEMANTIC.SETEX", "SETEX"):
            if len(args) < 4:
                return RESPSerializer.error(f"wrong number of arguments for '{cmd}'")
            key = args[1]
            try:
                seconds = int(args[2])
            except ValueError:
                return RESPSerializer.error("value is not an integer or out of range")
            val = args[3]

            vector = self.embedder.embed(key)
            self.storage.set(key, val, vector, ttl=seconds)
            return RESPSerializer.ok()

        # 4. Set TTL on existing key: EXPIRE <key> <seconds>
        if cmd == "EXPIRE":
            if len(args) < 3:
                return RESPSerializer.error(f"wrong number of arguments for '{cmd}'")
            key = args[1]
            try:
                seconds = float(args[2])
            except ValueError:
                return RESPSerializer.error("value is not a valid number")
            success = self.storage.expire(key, seconds)
            return RESPSerializer.integer(1 if success else 0)

        # 5. Check TTL: TTL <key>
        if cmd == "TTL":
            if len(args) < 2:
                return RESPSerializer.error(f"wrong number of arguments for '{cmd}'")
            remaining = self.storage.ttl(args[1])
            return RESPSerializer.integer(remaining)

        # 6. Retrieve: SEMANTIC.GET <key> (fast path bypasses vector embedding on exact match)
        if cmd in ("SEMANTIC.GET", "GET"):
            if len(args) < 2:
                return RESPSerializer.error(f"wrong number of arguments for '{cmd}'")
            key = args[1]

            # StorageManager checks L1 and L2 exact match in O(1) before invoking embedder
            result = self.storage.get(key, embed_fn=self.embedder.embed)

            if result is None:
                return RESPSerializer.bulk_string(None)  # $-1\r\n = Cache Miss!
            return RESPSerializer.bulk_string(result.value)

        # 7. Invalidate / Delete: DEL <key> or SEMANTIC.DEL <key>
        if cmd in ("SEMANTIC.DEL", "DEL"):
            if len(args) < 2:
                return RESPSerializer.error(f"wrong number of arguments for '{cmd}'")
            deleted = self.storage.delete(args[1])
            return RESPSerializer.integer(1 if deleted else 0)

        # 8. Tag Invalidation: TAG.INVALIDATE <tag>
        if cmd == "TAG.INVALIDATE":
            if len(args) < 2:
                return RESPSerializer.error(f"wrong number of arguments for '{cmd}'")
            count = self.storage.invalidate_tag(args[1])
            return RESPSerializer.integer(count)

        # 9. Existence check: EXISTS <key>
        if cmd == "EXISTS":
            if len(args) < 2:
                return RESPSerializer.error(f"wrong number of arguments for '{cmd}'")
            exists = args[1] in self.storage
            return RESPSerializer.integer(1 if exists else 0)

        # 10. Database size: DBSIZE
        if cmd == "DBSIZE":
            return RESPSerializer.integer(len(self.storage))

        # 11. Metrics: STATS
        if cmd == "STATS":
            stats_dict = self.storage.stats()
            stats_json = json.dumps(stats_dict, indent=2)
            return RESPSerializer.bulk_string(stats_json)

        # 12. Compact L2 disk storage: COMPACT
        if cmd == "COMPACT":
            reclaimed = self.storage.compact()
            return RESPSerializer.integer(reclaimed)

        # 13. Clear cache: FLUSHDB / FLUSHALL
        if cmd in ("FLUSHDB", "FLUSHALL"):
            self.storage.clear()
            return RESPSerializer.ok()

        # 14. Quit: QUIT
        if cmd == "QUIT":
            return RESPSerializer.ok()

        # Unknown command
        return RESPSerializer.error(f"unknown command '{args[0]}'")


def main() -> None:
    """Command-line interface entry point for starting the standalone cache server."""
    parser = argparse.ArgumentParser(description="Tiered Semantic Cache Standalone Daemon")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=6380, help="TCP port to listen on")
    parser.add_argument("--ram-capacity", type=int, default=1000, help="Max items in L1 RAM")
    parser.add_argument("--threshold", type=float, default=0.70, help="Similarity match threshold (0.0 - 1.0)")
    parser.add_argument("--dim", type=int, default=384, help="Embedding vector dimension")
    parser.add_argument("--disk-path", type=str, default="cache.db", help="Path for L2 disk storage file")

    cli_args = parser.parse_args()

    cfg = CacheConfig(
        ram_capacity=cli_args.ram_capacity,
        similarity_threshold=cli_args.threshold,
        disk_path=Path(cli_args.disk_path),
        vector_dim=cli_args.dim,
        port=cli_args.port,
        host=cli_args.host,
    )

    server = SemanticCacheServer(config=cfg)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("\n[*] Server shutdown cleanly.")


if __name__ == "__main__":
    main()
