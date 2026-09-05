"""
Phase 4 Verification Suite: TTL Lifecycle, Namespaces, Tag Invalidation & Client SDK
"""

import asyncio
import time
from pathlib import Path
import pytest

from semantic_cache import (
    TieredSemanticCache,
    CacheConfig,
    SemanticCacheClient,
)
from semantic_cache.server import SemanticCacheServer


def test_ttl_passive_expiration(tmp_path):
    """Test that records automatically expire passively on read."""
    config = CacheConfig(
        ram_capacity=5,
        disk_path=tmp_path / "test_ttl_passive.bin",
        enable_active_sweep=False,
    )
    with TieredSemanticCache(config) as cache:
        cache.set("quick_key", "quick_val", ttl=1)  # 1 second TTL
        cache.set("immortal_key", "forever_val")     # No TTL

        # Immediate read
        res = cache.get("quick_key")
        assert res is not None and res.value == "quick_val"
        assert res.ttl >= 0

        res_immortal = cache.get("immortal_key")
        assert res_immortal is not None and res_immortal.ttl == -1

        # Wait for expiration
        time.sleep(1.1)

        # Passive expiration should trigger on read
        assert cache.get("quick_key") is None
        assert "quick_key" not in cache
        assert cache.get("immortal_key") is not None


def test_ttl_l2_disk_expiration(tmp_path):
    """Test that expired records in persistent L2 disk are passively purged."""
    config = CacheConfig(
        ram_capacity=2,
        disk_path=tmp_path / "test_ttl_l2.bin",
        enable_active_sweep=False,
    )
    with TieredSemanticCache(config) as cache:
        cache.set("q1", "a1", ttl=1)
        cache.set("q2", "a2")
        cache.set("q3", "a3")  # q1 evicted to L2 Disk!

        assert "q1" in cache.storage.l2

        # Sleep past expiration
        time.sleep(1.1)

        # Attempt to get from L2
        assert cache.get("q1") is None
        assert "q1" not in cache.storage.l2


def test_tag_invalidation(tmp_path):
    """Test group invalidation using tags."""
    config = CacheConfig(disk_path=tmp_path / "test_tags.bin")
    with TieredSemanticCache(config) as cache:
        cache.set("q_tax", "tax advice", tags=["finance", "legal"])
        cache.set("q_stock", "stock quotes", tags=["finance"])
        cache.set("q_code", "python syntax", tags=["tech"])

        assert len(cache) == 3

        # Invalidate tag 'finance'
        purged = cache.invalidate_tag("finance")
        assert purged == 2

        assert cache.get("q_tax") is None
        assert cache.get("q_stock") is None
        assert cache.get("q_code") is not None
        assert len(cache) == 1


def test_namespace_isolation(tmp_path):
    """Test multi-tenant namespace isolation."""
    config = CacheConfig(disk_path=tmp_path / "test_namespaces.bin")
    with TieredSemanticCache(config) as cache:
        tenant_1 = cache.namespace("tenant_1")
        tenant_2 = cache.namespace("tenant_2")

        tenant_1.set("settings", "config_1", tags=["system"])
        tenant_2.set("settings", "config_2", tags=["system"])

        # Same key name returns isolated values
        assert tenant_1.get("settings").value == "config_1"
        assert tenant_2.get("settings").value == "config_2"

        # Invalidate tag in tenant_1 only
        tenant_1.invalidate_tag("system")
        assert tenant_1.get("settings") is None
        assert tenant_2.get("settings") is not None


def test_ttl_active_sweeper(tmp_path):
    """Test background active expiration thread sweeps dead records."""
    config = CacheConfig(
        ram_capacity=10,
        disk_path=tmp_path / "test_sweeper.bin",
        enable_active_sweep=True,
        sweep_interval_sec=0.2,
    )
    with TieredSemanticCache(config) as cache:
        for i in range(5):
            cache.set(f"temp_{i}", f"val_{i}", ttl=1)

        assert len(cache) == 5

        # Sleep past TTL and let background sweeper run
        time.sleep(1.3)

        assert len(cache) == 0
        assert cache.stats()["expired_purges"] >= 5


def test_client_tcp_and_fallback(tmp_path):
    """Test SemanticCacheClient over real TCP socket and fallback mode."""
    import threading

    db_path = tmp_path / "test_client_srv.bin"
    cfg = CacheConfig(port=6392, disk_path=db_path)
    server = SemanticCacheServer(config=cfg)

    # Run server in background thread with running event loop
    loop = asyncio.new_event_loop()
    started = threading.Event()

    def run_srv():
        asyncio.set_event_loop(loop)
        loop.run_until_complete(server.start())
        started.set()
        loop.run_forever()

    server_thread = threading.Thread(target=run_srv, daemon=True)
    server_thread.start()
    started.wait(timeout=3.0)

    try:
        with SemanticCacheClient(port=6392) as client:
            assert client.ping() == "PONG"

            # SET & GET
            assert client.set("how to code in python", "read python docs", ttl=5) is True
            ans = client.get("how to code in python")
            assert ans == "read python docs"

            # Check DBSIZE and TTL
            assert client.dbsize() == 1
            assert client.ttl("how to code in python") >= 0

            # DEL
            assert client.delete("how to code in python") is True
            assert client.get("how to code in python") is None

    finally:
        async def shutdown():
            await server.stop()
            loop.stop()

        loop.call_soon_threadsafe(lambda: asyncio.ensure_future(shutdown(), loop=loop))
        server_thread.join(timeout=2.0)

    # Server is offline now. Client with fallback_cache should still work seamlessly!
    local_cfg = CacheConfig(disk_path=tmp_path / "test_client_fallback.bin")
    with TieredSemanticCache(local_cfg) as fallback:
        client_fallback = SemanticCacheClient(port=6392, fallback_cache=fallback)
        client_fallback.set("offline_query", "offline_ans")
        assert client_fallback.get("offline_query") == "offline_ans"
        assert client_fallback.dbsize() == 1
        client_fallback.close()


def test_auto_compaction_and_waste_stats(tmp_path):
    """PR-6: Test disk waste tracking and automatic compaction when waste exceeds threshold."""
    db_path = tmp_path / "test_waste.bin"
    # Set ram_capacity=2 and auto_compact_waste_ratio=0.3
    config = CacheConfig(
        ram_capacity=2,
        disk_path=db_path,
        auto_compact_waste_ratio=0.3,
        enable_active_sweep=False,
    )
    with TieredSemanticCache(config) as cache:
        # Push items to disk (evict from RAM)
        cache.set("k1", "v1" * 100)
        cache.set("k2", "v2" * 100)
        cache.set("k3", "v3" * 100)
        cache.set("k4", "v4" * 100)

        assert len(cache.storage.l2) >= 2
        total_size, wasted, ratio = cache.storage.l2.waste_stats()
        assert total_size > 0
        assert ratio == 0.0

        # Repeatedly update k1 with large payloads to create dead space on disk
        for i in range(100):
            cache.storage.l2.append("k1", f"updated_val_{i}".encode() * 800, [0.1] * config.vector_dim)

        total_size, wasted, ratio = cache.storage.l2.waste_stats()
        assert wasted > 65536
        assert ratio > 0.5

        # Trigger sweep -> auto-compact should run because wasted > 64KB and ratio > 0.3
        cache.sweep_expired()

        # Check waste after compaction
        new_total, new_wasted, new_ratio = cache.storage.l2.waste_stats()
        assert new_wasted == 0
        assert new_total < total_size


def test_sweep_error_isolation(tmp_path):
    """SEC-5: Test that an exception during sweep in one tier does not stop the other tier."""
    config = CacheConfig(
        ram_capacity=5,
        disk_path=tmp_path / "test_sweep_err.bin",
        enable_active_sweep=False,
    )
    with TieredSemanticCache(config) as cache:
        cache.set("ram_key", "ram_val", ttl=1)
        time.sleep(1.1)

        # Mock l2.sweep_expired to raise an exception
        def failing_sweep():
            raise RuntimeError("Simulated L2 sweep failure")

        cache.storage.l2.sweep_expired = failing_sweep

        # sweep_expired should not crash and L1 should still be cleaned
        total = cache.sweep_expired()
        assert total == 1
        assert "ram_key" not in cache


def test_optional_index_file_lifecycle(tmp_path):
    """TC-3: Test fast index file creation, instant startup reload, and clean state recovery."""
    db_path = tmp_path / "test_index_lifecycle.db"
    idx_path = tmp_path / "test_index_lifecycle.db.idx"

    # 1. Start with enable_index_file=False -> no .idx created
    cfg_disabled = CacheConfig(disk_path=db_path, enable_index_file=False)
    with TieredSemanticCache(cfg_disabled) as cache:
        cache.set("foo", "bar")
    assert not idx_path.exists()

    # 2. Start with enable_index_file=True -> .idx created on close
    cfg_enabled = CacheConfig(
        ram_capacity=2,
        disk_path=db_path,
        enable_index_file=True,
        enable_active_sweep=False,
    )
    with TieredSemanticCache(cfg_enabled) as cache:
        cache.set("item1", "value1", ttl=60)
        cache.set("item2", "value2")
        cache.set("item3", "value3")  # item1 evicted to L2 disk

    assert idx_path.exists()
    assert idx_path.stat().st_size > 0

    # 3. Reopen with enable_index_file=True -> fast reload from index
    with TieredSemanticCache(cfg_enabled) as cache2:
        # L2 disk cache loaded from index
        assert "item1" in cache2.storage.l2
        rec = cache2.get("item1")
        assert rec is not None and rec.value == "value1"
        assert rec.ttl > 0

        # Semantic match should work from restored matrix
        res = cache2.get("item1")
        assert res is not None
        assert res.matched_key == "item1"


def test_index_file_tail_scan_and_corruption_fallback(tmp_path):
    """TC-3: Test tail fast-forwarding after crash and graceful fallback on corrupted index."""
    db_path = tmp_path / "test_tail_recovery.db"
    idx_path = tmp_path / "test_tail_recovery.db.idx"

    cfg = CacheConfig(ram_capacity=1, disk_path=db_path, enable_index_file=True)
    with TieredSemanticCache(cfg) as cache:
        cache.set("first", "val1")
        cache.set("second", "val2")

    assert idx_path.exists()

    # Simulate an unindexed append directly to the data file (e.g. crash before index save)
    l2 = cache.storage.l2
    # Reopen L2 directly to append a record without updating index file
    from semantic_cache.storage.l2_disk import L2DiskCache
    l2_direct = L2DiskCache(file_path=db_path, dim=cfg.vector_dim, enable_index_file=False)
    l2_direct.append("unindexed_key", b"unindexed_val", [0.2] * cfg.vector_dim)
    l2_direct.close()

    # Reopening with index file enabled should detect tail bytes and fast-forward
    with TieredSemanticCache(cfg) as cache2:
        assert "unindexed_key" in cache2.storage.l2
        rec = cache2.get("unindexed_key")
        assert rec is not None and rec.value == "unindexed_val"

    # Now corrupt the index file (flip bytes in header/CRC)
    with open(idx_path, "r+b") as f:
        f.seek(10)
        f.write(b"\xff\xff\xff\xff")

    # Cache should detect corrupt CRC, fall back to full scan, and still recover everything
    with TieredSemanticCache(cfg) as cache3:
        assert "first" in cache3.storage.l2 or "first" in cache3.storage.l1
        assert "unindexed_key" in cache3.storage.l2

