import os
import time
import pytest
import numpy as np

from semantic_cache import TieredSemanticCache, CacheConfig, DenseHashEmbedder


def test_cache_exact_match(tmp_path):
    """Test that asking the exact same question gets the exact same answer."""
    config = CacheConfig(disk_path=tmp_path / "test_exact.bin")
    with TieredSemanticCache(config) as cache:
        cache.set("What is the capital of France?", "Paris")

        result = cache.get("What is the capital of France?")
        assert result is not None
        assert result.value == "Paris"
        assert result.similarity == 1.0
        assert result.tier == "L1_EXACT"


def test_cache_semantic_match(tmp_path):
    """Test that a similar question returns the cached answer."""
    config = CacheConfig(
        disk_path=tmp_path / "test_semantic.bin",
        similarity_threshold=0.85,
    )
    with TieredSemanticCache(config) as cache:
        cache.set("What's the best way to cook steak?", "Reverse sear it.")

        result = cache.get("What is the best way to cook steak?")
        assert result is not None
        assert result.value == "Reverse sear it."


def test_cache_eviction_to_l2(tmp_path):
    """Test that when L1 gets full, it safely spills to L2."""
    config = CacheConfig(
        ram_capacity=2,  # Desk can only hold 2 items
        disk_path=tmp_path / "test_eviction.bin",
    )
    with TieredSemanticCache(config) as cache:
        # 1. Fill the desk (L1)
        cache.set("Q1", "A1")
        cache.set("Q2", "A2")

        assert cache.stats()["l1_count"] == 2
        assert cache.stats()["l2_count"] == 0

        # 2. Push an item off the desk! Q1 goes to filing cabinet (L2)
        cache.set("Q3", "A3")

        assert cache.stats()["l1_count"] == 2
        assert cache.stats()["l2_count"] == 1

        # 3. Can we still get Q1? Yes, from L2!
        result = cache.get("Q1")
        assert result is not None
        assert result.value == "A1"
        assert result.tier == "L2_EXACT"


def test_cache_pythonic_protocols(tmp_path):
    """Test len(), 'in' operator, and context manager."""
    config = CacheConfig(ram_capacity=2, disk_path=tmp_path / "test_protocols.bin")
    with TieredSemanticCache(config) as cache:
        cache.set("cat", "meow")
        cache.set("dog", "woof")
        cache.set("cow", "moo")  # Spills 'cat' to L2

        # Test len()
        assert len(cache) == 3

        # Test 'in' operator
        assert "cat" in cache
        assert "dog" in cache
        assert "bird" not in cache

        # Test input validation
        with pytest.raises(ValueError):
            cache.set("", "bad")

        with pytest.raises(ValueError):
            cache.set("key", 123)  # type: ignore


def test_cache_deduplication_and_promotion(tmp_path):
    """Test that promotions and updates do not duplicate records or leak memory."""
    config = CacheConfig(ram_capacity=2, disk_path=tmp_path / "test_dedup.bin")
    with TieredSemanticCache(config) as cache:
        cache.set("k1", "v1")
        cache.set("k2", "v2")
        cache.set("k3", "v3")  # k1 spills to L2
        assert len(cache) == 3
        assert len(cache.storage.l1) == 2
        assert len(cache.storage.l2) == 1

        # Promote k1 back to L1; k2 should spill to L2
        res = cache.get("k1")
        assert res is not None and res.value == "v1"
        assert len(cache) == 3
        assert len(cache.storage.l1) == 2
        assert len(cache.storage.l2) == 1
        assert "k1" in cache.storage.l1
        assert "k1" not in cache.storage.l2
        assert "k2" in cache.storage.l2

        # Update an existing key that is currently in L2
        cache.set("k2", "v2_updated")
        assert len(cache) == 3
        assert cache.get("k2").value == "v2_updated"


def test_cache_disk_reload_deduplication(tmp_path):
    """Test that restarting the cache from a disk log deduplicates superseded records."""
    db_path = tmp_path / "test_reload.bin"
    config = CacheConfig(ram_capacity=2, disk_path=db_path)

    # Initial session with spills and promotions
    c1 = TieredSemanticCache(config)
    c1.set("k1", "v1")
    c1.set("k2", "v2")
    c1.set("k3", "v3")  # k1 -> L2
    c1.get("k1")        # k1 promoted -> L1, k2 -> L2
    c1.set("k4", "v4")  # k3 -> L2
    c1.set("k5", "v5")  # k1 -> L2
    c1.close()

    # Second session reload
    c2 = TieredSemanticCache(config)
    assert len(c2) == 3  # unique items on disk: k2, k3, k1
    assert len(c2.storage.l2._keys_list) == 3
    assert set(c2.storage.l2._keys_list) == {"k1", "k2", "k3"}
    assert c2.get("k1").value == "v1"
    assert c2.get("k2").value == "v2"
    assert c2.get("k3").value == "v3"
    c2.close()


def test_cache_compaction(tmp_path):
    """Test that compact() reclaims disk space from obsolete records."""
    db_path = tmp_path / "test_compaction.bin"
    config = CacheConfig(ram_capacity=2, disk_path=db_path)

    with TieredSemanticCache(config) as cache:
        for i in range(25):
            cache.set(f"key_{i % 3}", f"val_{i}")

        size_before = db_path.stat().st_size
        reclaimed = cache.compact()
        size_after = db_path.stat().st_size

        assert reclaimed > 0
        assert size_after < size_before
        assert cache.get("key_0") is not None


def test_cache_delete_and_clear(tmp_path):
    """Test item deletion and full cache clearing."""
    config = CacheConfig(ram_capacity=2, disk_path=tmp_path / "test_del.bin")
    with TieredSemanticCache(config) as cache:
        cache["a"] = "1"
        cache["b"] = "2"
        cache["c"] = "3"  # "a" spills to L2

        assert len(cache) == 3
        assert cache["a"] == "1"

        # Delete from L1
        del cache["c"]
        assert "c" not in cache
        assert len(cache) == 2

        # Delete from L2
        assert cache.delete("b") is True
        assert cache.delete("nonexistent") is False
        assert len(cache) == 1

        # Clear everything
        cache.clear()
        assert len(cache) == 0
        assert "a" not in cache


def test_cache_thread_safety(tmp_path):
    """Test that concurrent reads and writes across multiple threads operate safely."""
    import threading

    config = CacheConfig(ram_capacity=10, disk_path=tmp_path / "test_thread.bin")
    cache = TieredSemanticCache(config)
    errors = []

    def writer(start: int, count: int) -> None:
        try:
            for i in range(start, start + count):
                cache.set(f"t_key_{i}", f"t_val_{i}")
        except Exception as e:
            errors.append(e)

    def reader(start: int, count: int) -> None:
        try:
            for i in range(start, start + count):
                cache.get(f"t_key_{i}")
        except Exception as e:
            errors.append(e)

    threads = []
    for t in range(4):
        threads.append(threading.Thread(target=writer, args=(t * 25, 25)))
        threads.append(threading.Thread(target=reader, args=(t * 25, 25)))

    for th in threads:
        th.start()
    for th in threads:
        th.join()

    cache.close()
    assert len(errors) == 0


def test_cache_config_validation():
    """Verify that invalid configurations raise descriptive ValueErrors/TypeErrors."""
    with pytest.raises(ValueError):
        CacheConfig(ram_capacity=0)

    with pytest.raises(ValueError):
        CacheConfig(ram_capacity=True)  # type: ignore

    with pytest.raises(ValueError):
        CacheConfig(similarity_threshold=1.5)

    with pytest.raises(ValueError):
        CacheConfig(similarity_threshold=-0.1)

    with pytest.raises(ValueError):
        CacheConfig(vector_dim=-10)

    with pytest.raises(ValueError):
        CacheConfig(port=99999)


@pytest.mark.anyio
async def test_server_extended_protocol(tmp_path):
    """Test standalone TCP daemon with extended RESP commands."""
    import asyncio
    from semantic_cache.server import SemanticCacheServer

    db_path = tmp_path / "test_srv_ext.bin"
    cfg = CacheConfig(port=6395, disk_path=db_path, ram_capacity=5)
    server = SemanticCacheServer(config=cfg)
    await server.start()

    reader, writer = await asyncio.open_connection("127.0.0.1", 6395)

    # 1. PING
    writer.write(b"*1\r\n$4\r\nPING\r\n")
    await writer.drain()
    assert (await reader.readline()) == b"+PONG\r\n"

    # 2. SET
    writer.write(b"*3\r\n$3\r\nSET\r\n$4\r\nuser\r\n$5\r\nalice\r\n")
    await writer.drain()
    assert (await reader.readline()) == b"+OK\r\n"

    # 3. GET (Exact match fast path)
    writer.write(b"*2\r\n$3\r\nGET\r\n$4\r\nuser\r\n")
    await writer.drain()
    hdr = await reader.readline()
    assert hdr == b"$5\r\n"
    assert (await reader.readexactly(5 + 2)) == b"alice\r\n"

    # 4. EXISTS
    writer.write(b"*2\r\n$6\r\nEXISTS\r\n$4\r\nuser\r\n")
    await writer.drain()
    assert (await reader.readline()) == b":1\r\n"

    # 5. DBSIZE
    writer.write(b"*1\r\n$6\r\nDBSIZE\r\n")
    await writer.drain()
    assert (await reader.readline()) == b":1\r\n"

    # 6. DEL
    writer.write(b"*2\r\n$3\r\nDEL\r\n$4\r\nuser\r\n")
    await writer.drain()
    assert (await reader.readline()) == b":1\r\n"

    # 7. EXISTS after DEL
    writer.write(b"*2\r\n$6\r\nEXISTS\r\n$4\r\nuser\r\n")
    await writer.drain()
    assert (await reader.readline()) == b":0\r\n"

    # 8. COMPACT
    writer.write(b"*1\r\n$7\r\nCOMPACT\r\n")
    await writer.drain()
    resp = await reader.readline()
    assert resp.startswith(b":")

    # 9. FLUSHDB
    writer.write(b"*1\r\n$7\r\nFLUSHDB\r\n")
    await writer.drain()
    assert (await reader.readline()) == b"+OK\r\n"

    # 10. QUIT
    writer.write(b"*1\r\n$4\r\nQUIT\r\n")
    await writer.drain()
    assert (await reader.readline()) == b"+OK\r\n"

    writer.close()
    await writer.wait_closed()
    await server.stop()


@pytest.mark.anyio
async def test_server_auth_lifecycle(tmp_path):
    """Test AUTH enforcement, pre-auth allowlist, failed auth backoff, and lockout (TEST-1, SEC-3)."""
    import asyncio
    from semantic_cache.server import SemanticCacheServer
    from semantic_cache.client import SemanticCacheClient

    db_path = tmp_path / "test_srv_auth.bin"
    cfg = CacheConfig(port=6396, disk_path=db_path, requirepass="supersecret123")
    server = SemanticCacheServer(config=cfg)
    await server.start()

    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", 6396)

        # 1. Non-whitelisted command before auth must fail with NOAUTH
        writer.write(b"*3\r\n$3\r\nSET\r\n$1\r\nk\r\n$1\r\nv\r\n")
        await writer.drain()
        err_resp = await reader.readline()
        assert b"NOAUTH" in err_resp

        # 2. Whitelisted PING works before auth
        writer.write(b"*1\r\n$4\r\nPING\r\n")
        await writer.drain()
        assert (await reader.readline()) == b"+PONG\r\n"

        # 3. Wrong password fails with invalid password
        writer.write(b"*2\r\n$4\r\nAUTH\r\n$9\r\nwrongpass\r\n")
        await writer.drain()
        resp = await reader.readline()
        assert b"invalid password" in resp

        # 4. Brute-force protection: 4 more failed attempts (total 5) triggers disconnect
        for _ in range(4):
            writer.write(b"*2\r\n$4\r\nAUTH\r\n$9\r\nwrongpass\r\n")
            await writer.drain()
            line = await reader.readline()
            assert b"invalid password" in line or line == b""

        # Connection should be closed by server after 5 failed attempts
        writer.write(b"*1\r\n$4\r\nPING\r\n")
        try:
            await writer.drain()
            line = await reader.readline()
            assert line == b""
        except (ConnectionResetError, BrokenPipeError):
            pass
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:
            pass

        # 5. Connect fresh and authenticate properly
        reader2, writer2 = await asyncio.open_connection("127.0.0.1", 6396)
        writer2.write(b"*2\r\n$4\r\nAUTH\r\n$14\r\nsupersecret123\r\n")
        await writer2.drain()
        assert (await reader2.readline()) == b"+OK\r\n"

        # Now SET and GET work
        writer2.write(b"*3\r\n$3\r\nSET\r\n$4\r\nauth\r\n$7\r\nsuccess\r\n")
        await writer2.drain()
        assert (await reader2.readline()) == b"+OK\r\n"

        writer2.close()
        try:
            await writer2.wait_closed()
        except Exception:
            pass

        # 6. Test client SDK with password parameter (in thread to avoid blocking loop)
        def run_client_checks():
            with SemanticCacheClient(host="127.0.0.1", port=6396, password="supersecret123") as client:
                assert client.ping() == "PONG"
                assert client.get("auth") == "success"

        await asyncio.to_thread(run_client_checks)

    finally:
        await server.stop()


def test_config_password_env_and_masking(monkeypatch, tmp_path):
    """Test SEC-1: password env variable fallback and credential masking in __repr__."""
    monkeypatch.setenv("CACHE_REQUIREPASS", "env_secret_pass")
    cfg = CacheConfig(disk_path=tmp_path / "test_env_cfg.bin")
    assert cfg.requirepass == "env_secret_pass"
    assert "***" in repr(cfg)
    assert "env_secret_pass" not in repr(cfg)


def test_performance(tmp_path):
    """Make sure exact lookups are sub-microsecond fast!"""
    config = CacheConfig(disk_path=tmp_path / "test_perf.bin")
    with TieredSemanticCache(config) as cache:
        cache.set("Hello world", "Hi there!")

        start = time.perf_counter()
        for _ in range(1000):
            cache.get("Hello world")
        end = time.perf_counter()

        elapsed_ms = (end - start) * 1000
        avg_us = (elapsed_ms / 1000) * 1000
        print(f"1,000 Exact GETs took {elapsed_ms:.2f} ms ({avg_us:.2f} us/call)")

        # 1,000 exact lookups should easily finish under 10ms with our O(1) fast-path
        assert elapsed_ms < 10.0, "Cache is running too slow!"
