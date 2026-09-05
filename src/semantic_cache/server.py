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
0. AUTH <password>                -> Authenticate before using any command.
1. PING [message]                -> Health check ("+PONG").
2. SEMANTIC.SET <key> <val>      -> Saves question & answer across RAM/Disk ("+OK").
3. SEMANTIC.SETEX <k> <sec> <v>  -> Saves with an expiration countdown in seconds.
4. SEMANTIC.GET <key>            -> Searches for exact text OR close meaning match.
5. EXPIRE <key> <seconds>        -> Sets or updates an expiration timer on a saved item.
6. TTL <key>                     -> Returns remaining seconds before expiration.
7. DEL <key>                     -> Deletes an item from both RAM and Disk.
8. TAG.INVALIDATE <tag>          -> Deletes all items associated with a given tag.
9. EXISTS <key>                  -> Checks if an item exists (1 if yes, 0 if no).
10. DBSIZE                       -> Returns total count of all cached items.
11. STATS                        -> Returns health metrics (hits, misses, counts) as JSON.
12. COMPACT                      -> Cleans up dead space on disk to reclaim storage.
13. FLUSHDB / FLUSHALL           -> Wipes the entire cache clean.
14. QUIT                         -> Closes the client connection.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import ssl
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from semantic_cache.config import CacheConfig
from semantic_cache.embedder import BaseEmbedder, DenseHashEmbedder
from semantic_cache.protocol import RESPParser, RESPSerializer
from semantic_cache.storage.manager import StorageManager

logger = logging.getLogger("semantic_cache.server")

# Commands allowed before authentication
_PRE_AUTH_COMMANDS = frozenset({"AUTH", "PING", "QUIT"})


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

        # Connection limiter: prevents OOM from too many simultaneous clients
        self._conn_semaphore = asyncio.Semaphore(self.config.max_connections)
        self._active_connections: int = 0

        # Thread pool for offloading blocking cache operations off the event loop
        self._executor = ThreadPoolExecutor(
            max_workers=min(8, (os.cpu_count() or 4) + 2),
            thread_name_prefix="tsc-worker",
        )

        logger.info(
            "Server initialised: ram_capacity=%d, threshold=%.2f, dim=%d, auth=%s",
            self.config.ram_capacity,
            self.config.similarity_threshold,
            self.config.vector_dim,
            "enabled" if self.config.requirepass else "disabled",
        )

    async def start(self) -> None:
        """Start listening for incoming client connections with optional TLS."""
        ssl_ctx: Optional[ssl.SSLContext] = None
        if self.config.ssl_certfile is not None:
            ssl_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
            ssl_ctx.load_cert_chain(
                certfile=str(self.config.ssl_certfile),
                keyfile=str(self.config.ssl_keyfile) if self.config.ssl_keyfile else None,
            )
            if self.config.ssl_ca_certs:
                ssl_ctx.load_verify_locations(cafile=str(self.config.ssl_ca_certs))
                ssl_ctx.verify_mode = ssl.CERT_REQUIRED
            logger.info("TLS enabled on server with certificate: %s", self.config.ssl_certfile)

        self._server = await asyncio.start_server(
            self._handle_client,
            host=self.config.host,
            port=self.config.port,
            ssl=ssl_ctx,
        )
        self._running = True
        logger.info(
            "Listening on %s:%d (TLS=%s)",
            self.config.host,
            self.config.port,
            "yes" if ssl_ctx else "no",
        )

    async def stop(self) -> None:
        """Gracefully shut down the server and close storage files."""
        logger.info("Shutting down server...")
        self._running = False
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
        self._executor.shutdown(wait=False)
        self.storage.close()
        logger.info("Server shutdown complete.")

    async def serve_forever(self) -> None:
        """Start the server and run until cancelled."""
        await self.start()
        print(f"[*] Semantic Cache Server running on {self.config.host}:{self.config.port}")
        print(f"[*] RAM Capacity: {self.config.ram_capacity} | Threshold: {self.config.similarity_threshold}")
        if self.config.requirepass:
            print("[*] AUTH: enabled (password required)")
        else:
            print("[*] AUTH: disabled (no password set)")

        # Register graceful shutdown on SIGTERM (Docker/Kubernetes sends this)
        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            loop.add_signal_handler(signal.SIGTERM, lambda: asyncio.ensure_future(self.stop()))

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
        # Check connection limit
        if not self._conn_semaphore._value:
            logger.warning("Connection rejected: max_connections (%d) reached", self.config.max_connections)
            writer.write(RESPSerializer.error("max connections reached"))
            writer.close()
            return

        async with self._conn_semaphore:
            self._active_connections += 1
            peer = writer.get_extra_info("peername", ("unknown", 0))
            logger.debug("Client connected: %s", peer)

            # Per-connection auth state
            authenticated = self.config.requirepass is None  # No password = auto-authenticated
            failed_auth_attempts = 0

            try:
                while self._running:
                    args = await RESPParser.read_command(reader)
                    if args is None:
                        break  # Client disconnected or malformed payload

                    cmd = args[0].upper() if args else ""

                    # Auth gate: if password is required, only allow AUTH/PING/QUIT before login
                    if not authenticated and cmd not in _PRE_AUTH_COMMANDS:
                        writer.write(RESPSerializer.error("NOAUTH Authentication required"))
                        await writer.drain()
                        continue

                    # Handle AUTH command inline (fast, no need for executor)
                    if cmd == "AUTH":
                        if self.config.requirepass is None:
                            writer.write(RESPSerializer.error("no password is set"))
                        elif len(args) < 2:
                            writer.write(RESPSerializer.error("wrong number of arguments for 'AUTH'"))
                        elif args[1] == self.config.requirepass:
                            authenticated = True
                            failed_auth_attempts = 0
                            logger.info("Client %s authenticated successfully", peer)
                            writer.write(RESPSerializer.ok())
                        else:
                            failed_auth_attempts += 1
                            logger.warning("Failed AUTH attempt #%d from %s", failed_auth_attempts, peer)
                            writer.write(RESPSerializer.error("invalid password"))
                            await writer.drain()
                            # Progressive exponential backoff against brute-force attacks (SEC-3)
                            backoff = min(1.0, 0.05 * (2 ** (failed_auth_attempts - 1)))
                            await asyncio.sleep(backoff)
                            if failed_auth_attempts >= 5:
                                logger.warning(
                                    "Max failed AUTH attempts (5) reached for %s; disconnecting client", peer
                                )
                                break
                            continue
                        await writer.drain()
                        continue

                    # Offload blocking cache work to thread pool (LAT-1)
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        self._executor, self._execute_command, args
                    )
                    writer.write(response)
                    await writer.drain()

                    # If client requested QUIT, close connection
                    if cmd == "QUIT":
                        break

            except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError, ValueError):
                pass
            except Exception:
                logger.exception("Unexpected error handling client %s", peer)
            finally:
                self._active_connections -= 1
                logger.debug("Client disconnected: %s (active: %d)", peer, self._active_connections)
                writer.close()
                try:
                    await writer.wait_closed()
                except Exception:
                    pass

    def _execute_command(self, args: list[str]) -> bytes:
        """Route and execute a parsed command. Runs in thread pool."""
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
            logger.debug("SET key=%s (len=%d)", key[:50], len(val))
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
            logger.debug("SETEX key=%s ttl=%d", key[:50], seconds)
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
                logger.debug("GET miss: key=%s", key[:50])
                return RESPSerializer.bulk_string(None)  # $-1\r\n = Cache Miss!
            logger.debug("GET hit: key=%s tier=%s sim=%.3f", key[:50], result.tier, result.similarity)
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
            logger.debug("TAG.INVALIDATE tag=%s removed=%d", args[1], count)
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
            stats_dict["active_connections"] = self._active_connections
            stats_json = json.dumps(stats_dict, indent=2)
            return RESPSerializer.bulk_string(stats_json)

        # 12. Compact L2 disk storage: COMPACT
        if cmd == "COMPACT":
            logger.info("COMPACT started")
            reclaimed = self.storage.compact()
            logger.info("COMPACT finished: reclaimed %d bytes", reclaimed)
            return RESPSerializer.integer(reclaimed)

        # 13. Clear cache: FLUSHDB / FLUSHALL
        if cmd in ("FLUSHDB", "FLUSHALL"):
            self.storage.clear()
            logger.info("Cache flushed via %s", cmd)
            return RESPSerializer.ok()

        # 14. Quit: QUIT
        if cmd == "QUIT":
            return RESPSerializer.ok()

        # Unknown command
        logger.warning("Unknown command: %s", args[0])
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
    parser.add_argument("--requirepass", type=str, default=None, help="Require password for client connections")
    parser.add_argument("--max-connections", type=int, default=1000, help="Max simultaneous client connections")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging level")

    cli_args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, cli_args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = CacheConfig(
        ram_capacity=cli_args.ram_capacity,
        similarity_threshold=cli_args.threshold,
        disk_path=Path(cli_args.disk_path),
        vector_dim=cli_args.dim,
        port=cli_args.port,
        host=cli_args.host,
        requirepass=cli_args.requirepass,
        max_connections=cli_args.max_connections,
    )

    server = SemanticCacheServer(config=cfg)
    try:
        asyncio.run(server.serve_forever())
    except KeyboardInterrupt:
        print("\n[*] Server shutdown cleanly.")


if __name__ == "__main__":
    main()
