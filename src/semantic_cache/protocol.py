"""
Tiered Semantic Cache - RESP Wire Protocol Engine
=================================================

What is this file?
------------------
This file teaches our server how to speak RESP (REdis Serialization Protocol).
RESP is the international language spoken by Redis.
Because we speak standard RESP:
- Any programming language (Python, Node.js, Go, Rust, Java) can talk to us.
- You can even talk to this cache using the standard command-line tool:
  redis-cli -p 6380

The 5 Secret Symbols of RESP in Plain English:
----------------------------------------------
Every message sent over the wire starts with 1 special symbol:
* '+' = Simple String: Short happy message like "+OK\r\n" or "+PONG\r\n".
* '-' = Error: Something went wrong like "-ERR unknown command\r\n".
* ':' = Number (Integer): Numbers like ":100\r\n".
* '$' = Bulk String: Any text with its exact byte length:
        Example: "$5\r\nhello\r\n" (reads exactly 5 bytes for 'hello').
        Special: "$-1\r\n" means Null / Nil (our CACHE MISS!).
* '*' = Array: A list of items like "*2\r\n$4\r\nPING\r\n$2\r\nhi\r\n".
"""

from __future__ import annotations

import asyncio
from typing import Optional, Sequence, Union


MAX_ARGS = 1024
MAX_BULK_LEN = 64 * 1024 * 1024  # 64 MB max payload


class RESPSerializer:
    """Encodes Python data into standard Redis Serialization Protocol bytes."""

    @staticmethod
    def ok() -> bytes:
        """Standard success response (+OK)."""
        return b"+OK\r\n"

    @staticmethod
    def pong(message: Optional[str] = None) -> bytes:
        """Standard ping response (+PONG or echoed text)."""
        if message:
            return RESPSerializer.bulk_string(message)
        return b"+PONG\r\n"

    @staticmethod
    def simple_string(text: str) -> bytes:
        """Encode a short one-line status string (+text)."""
        clean = text.replace("\r", " ").replace("\n", " ")
        return f"+{clean}\r\n".encode("utf-8")

    @staticmethod
    def error(message: str) -> bytes:
        """Encode an error message (-ERR message)."""
        clean = message.replace("\r", " ").replace("\n", " ")
        return f"-ERR {clean}\r\n".encode("utf-8")

    @staticmethod
    def integer(value: int) -> bytes:
        """Encode a whole number (:number)."""
        return f":{int(value)}\r\n".encode("utf-8")

    @staticmethod
    def bulk_string(value: Optional[Union[str, bytes]]) -> bytes:
        """Encode arbitrary text or binary data. Returns '$-1\r\n' for None (Cache Miss)."""
        if value is None:
            return b"$-1\r\n"

        data = value.encode("utf-8") if isinstance(value, str) else value
        return f"${len(data)}\r\n".encode("utf-8") + data + b"\r\n"

    @staticmethod
    def array(items: Sequence[bytes]) -> bytes:
        """Encode a list of RESP elements (*count)."""
        return f"*{len(items)}\r\n".encode("utf-8") + b"".join(items)


class RESPParser:
    """Zero-allocation streaming parser reading RESP and inline commands from TCP socket."""

    @staticmethod
    async def read_command(reader: asyncio.StreamReader) -> Optional[list[str]]:
        """Read the next command from the client stream.

        Supports two formats:
        1. Standard RESP Array: *<count>\r\n$<len>\r\n<arg>\r\n...
        2. Plain Inline Command: PING\r\n or STATS\r\n (for telnet/nc).

        Returns:
            List of string arguments (e.g. ['SEMANTIC.GET', 'my query']),
            or None if client disconnected or invalid payload.
        """
        try:
            while True:
                line = await reader.readline()
                if not line:
                    return None  # Client disconnected

                # Strip trailing CRLF or LF
                clean = line.rstrip(b"\r\n")
                if not clean:
                    continue  # Ignore empty newline heartbeats
                break

            # 1. Handle standard RESP Array (*<count>\r\n)
            if clean.startswith(b"*"):
                count = int(clean[1:])
                if count <= 0 or count > MAX_ARGS:
                    return None

                args: list[str] = []
                for _ in range(count):
                    header = await reader.readline()
                    if not header or not header.startswith(b"$"):
                        return None

                    str_len = int(header[1:].rstrip(b"\r\n"))
                    if str_len < 0 or str_len > MAX_BULK_LEN:
                        return None

                    # Read exact data bytes + 2 bytes for \r\n
                    payload = await reader.readexactly(str_len + 2)
                    if payload[-2:] != b"\r\n":
                        return None
                    arg_bytes = payload[:-2]  # slice off \r\n
                    args.append(arg_bytes.decode("utf-8", errors="replace"))

                return args

            # 2. Handle plain text inline command (e.g. "PING" or "STATS")
            parts = clean.decode("utf-8", errors="replace").split()
            return parts if parts else None

        except (asyncio.IncompleteReadError, ValueError):
            return None
