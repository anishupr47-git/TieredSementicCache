"""
Phase 3 Verification Test Script: Wire Protocol & AsyncIO TCP Server
"""

import asyncio
from pathlib import Path
import sys

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from semantic_cache.config import CacheConfig
from semantic_cache.protocol import RESPSerializer, RESPParser
from semantic_cache.server import SemanticCacheServer

test_db = Path("test_server_phase3.db")


async def run_server_tests():
    print("=" * 70)
    print(">>> TESTING PHASE 3: WIRE PROTOCOL & ASYNCIO TCP SERVER <<<")
    print("=" * 70)

    # 1. Verify Serializer
    assert RESPSerializer.ok() == b"+OK\r\n"
    assert RESPSerializer.pong() == b"+PONG\r\n"
    assert RESPSerializer.pong("hi") == b"$2\r\nhi\r\n"
    assert RESPSerializer.error("bad") == b"-ERR bad\r\n"
    assert RESPSerializer.bulk_string(None) == b"$-1\r\n"
    assert RESPSerializer.bulk_string("hello") == b"$5\r\nhello\r\n"
    assert RESPSerializer.integer(42) == b":42\r\n"
    print("[OK] RESPSerializer checks PASSED!")

    # 2. Start AsyncIO Server on port 6389
    if test_db.exists():
        test_db.unlink()

    cfg = CacheConfig(
        port=6389,
        disk_path=test_db,
        similarity_threshold=0.70,
        ram_capacity=10,
    )
    server = SemanticCacheServer(config=cfg)
    await server.start()
    print("[OK] SemanticCacheServer listening on port 6389!")

    # 3. Connect real TCP client
    reader, writer = await asyncio.open_connection("127.0.0.1", 6389)
    print("[OK] Client connected to 127.0.0.1:6389!")

    # Test PING
    writer.write(b"PING\r\n")
    await writer.drain()
    resp = await reader.readline()
    assert resp == b"+PONG\r\n"
    print("[OK] PING -> +PONG verified!")

    # Test PING with message
    writer.write(b"*2\r\n$4\r\nPING\r\n$5\r\nhello\r\n")
    await writer.drain()
    hdr = await reader.readline()
    assert hdr == b"$5\r\n"
    data = await reader.readexactly(5 + 2)
    assert data == b"hello\r\n"
    print("[OK] PING hello -> $5\r\nhello\r\n verified!")

    # Test SEMANTIC.SET using RESP Array: SET "how to install python" "download from python.org"
    key = "how to install python"
    val = "download from python.org"
    cmd_set = (
        f"*3\r\n$12\r\nSEMANTIC.SET\r\n${len(key)}\r\n{key}\r\n${len(val)}\r\n{val}\r\n"
    ).encode("utf-8")
    writer.write(cmd_set)
    await writer.drain()
    resp = await reader.readline()
    assert resp == b"+OK\r\n"
    print("[OK] SEMANTIC.SET -> +OK verified!")

    # Test Exact SEMANTIC.GET
    cmd_get_exact = f"*2\r\n$12\r\nSEMANTIC.GET\r\n${len(key)}\r\n{key}\r\n".encode("utf-8")
    writer.write(cmd_get_exact)
    await writer.drain()
    hdr = await reader.readline()
    assert hdr == f"${len(val)}\r\n".encode("utf-8")
    data = await reader.readexactly(len(val) + 2)
    assert data == f"{val}\r\n".encode("utf-8")
    print("[OK] Exact SEMANTIC.GET hit verified!")

    # Test Semantic SEMANTIC.GET with rephrased query: "install python how"
    query_rephrased = "install python how"
    cmd_get_sem = f"*2\r\n$12\r\nSEMANTIC.GET\r\n${len(query_rephrased)}\r\n{query_rephrased}\r\n".encode("utf-8")
    writer.write(cmd_get_sem)
    await writer.drain()
    hdr = await reader.readline()
    assert hdr == f"${len(val)}\r\n".encode("utf-8")
    data = await reader.readexactly(len(val) + 2)
    assert data == f"{val}\r\n".encode("utf-8")
    print("[OK] Semantic SEMANTIC.GET hit across network socket verified!")

    # Test Cache Miss: "astronomy and mars space exploration"
    query_miss = "astronomy and mars space exploration"
    cmd_miss = f"*2\r\n$12\r\nSEMANTIC.GET\r\n${len(query_miss)}\r\n{query_miss}\r\n".encode("utf-8")
    writer.write(cmd_miss)
    await writer.drain()
    resp = await reader.readline()
    assert resp == b"$-1\r\n"  # Standard Redis Null!
    print("[OK] Semantic Cache Miss ($-1) verified!")

    # Test STATS command
    writer.write(b"STATS\r\n")
    await writer.drain()
    hdr = await reader.readline()
    assert hdr.startswith(b"$")
    stats_len = int(hdr[1:].strip())
    stats_raw = await reader.readexactly(stats_len + 2)
    assert b"total_hits" in stats_raw
    assert b"l1_hits" in stats_raw
    print("[OK] STATS command returned valid JSON metrics!")

    # Test QUIT
    writer.write(b"QUIT\r\n")
    await writer.drain()
    resp = await reader.readline()
    assert resp == b"+OK\r\n"
    writer.close()
    await writer.wait_closed()
    print("[OK] QUIT command verified!")

    # Stop server
    await server.stop()
    print("[OK] Server stopped cleanly!")

    if test_db.exists():
        test_db.unlink()
    print("=" * 70)
    print("🎉 ALL PHASE 3 WIRE PROTOCOL & TCP DAEMON TESTS PASSED 100%!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_server_tests())
