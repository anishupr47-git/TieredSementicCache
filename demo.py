"""
TieredSemanticCache - Complete Interactive Tour & Living Documentation
======================================================================

Run this file anytime:
    python demo.py

This file demonstrates everything you can do with `tiered-semantic-cache`:
  1. Standalone Text-to-Vector Embedder (DenseHashEmbedder)
  2. Basic Storing & Fast Retrieving (Exact vs Semantic match)
  3. Python Dictionary Syntax (`[]`, `in`, `len`, `del`)
  4. Time-To-Live (TTL) & Expiration Timers
  5. Category Tag Invalidation
  6. Multi-User Tenant Isolation (Ram vs Shyam)
  7. Disk Compaction & System Stats
  8. Web Frameworks Integration (FastAPI & Django)
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

# Windows terminal UTF-8 encoding support
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add 'src' folder to path for local runs
sys.path.insert(0, str(Path(__file__).parent / "src"))

import numpy as np
from semantic_cache import (
    CacheConfig,
    DenseHashEmbedder,
    NamespacedSemanticCache,
    TieredSemanticCache,
)


def section(title: str) -> None:
    print("\n" + "=" * 75)
    print(f">>> {title}")
    print("=" * 75)


def demo_standalone_embedder() -> None:
    section("1. STANDALONE VECTOR EMBEDDER (DenseHashEmbedder)")
    print("You can use DenseHashEmbedder completely standalone for your own AI projects,")
    print("search engines, or clustering without needing any external API keys!\n")

    embedder = DenseHashEmbedder(dim=384)
    text = "How do I install Python on Windows?"
    vector = embedder.embed(text)

    print(f"Input Text   : '{text}'")
    print(f"Vector Shape : {vector.shape} (384 numbers)")
    print(f"Sample Vector: {np.round(vector[:6], 4)} ...")
    print(f"Vector Length: {np.linalg.norm(vector):.6f} (Normalized to exactly 1.0!)")


def demo_basic_cache() -> None:
    section("2. BASIC CACHE: EXACT MATCH VS SEMANTIC MATCH")
    print("Demonstrates 1-step exact match (0ms) and fuzzy semantic match.\n")

    cache = TieredSemanticCache(
        config=CacheConfig(similarity_threshold=0.70, ram_capacity=50)
    )

    # Store a prompt and response
    cache.set(
        query="How do I install Python on Windows?",
        answer="Download the official Windows installer from python.org and run the setup.",
    )
    print("Stored: 'How do I install Python on Windows?'\n")

    # A. Exact Match (Sub-microsecond fast path)
    res_exact = cache.get("How do I install Python on Windows?")
    if res_exact:
        print("[A] Exact Query Match:")
        print(f"    Query      : 'How do I install Python on Windows?'")
        print(f"    Similarity : {res_exact.similarity:.4f} (100% Match!)")
        print(f"    Tier       : {res_exact.tier}")
        print(f"    Answer     : {res_exact.value[:60]}...\n")

    # B. Semantic Match (Different words, same meaning!)
    res_semantic = cache.get("install python windows")
    if res_semantic:
        print("[B] Semantic Meaning Match:")
        print(f"    Query      : 'install python windows'")
        print(f"    Similarity : {res_semantic.similarity:.4f} (High Match >= 0.70)")
        print(f"    Matched To : '{res_semantic.matched_key}'")
        print(f"    Tier       : {res_semantic.tier}")
        print(f"    Answer     : {res_semantic.value[:60]}...\n")

    # C. Unrelated Query (Cache Miss)
    res_miss = cache.get("Delicious recipe for chocolate chip cookies")
    print("[C] Unrelated Query Match:")
    print(f"    Query      : 'Delicious recipe for chocolate chip cookies'")
    print(f"    Result     : {'None (Cache Miss!)' if res_miss is None else res_miss.value}")

    cache.close()


def demo_dictionary_syntax() -> None:
    section("3. PYTHON DICTIONARY SYNTAX (`[]`, `in`, `len`, `del`)")
    print("You can use the cache just like an ordinary Python dictionary!\n")

    cache = TieredSemanticCache()

    # Subscription set: cache[key] = value
    cache["Who was Albert Einstein?"] = "A Nobel Prize-winning theoretical physicist."
    cache["What is the speed of light?"] = "Approximately 299,792 kilometers per second."

    # Subscription get: cache[key]
    print(f"cache['Who was Albert Einstein?'] -> {cache['Who was Albert Einstein?']}")

    # Membership check: key in cache
    print(f"'What is the speed of light?' in cache -> {'What is the speed of light?' in cache}")
    print(f"'Who is Isaac Newton?' in cache        -> {'Who is Isaac Newton?' in cache}")

    # Total length: len(cache)
    print(f"Total items in cache (len) -> {len(cache)}")

    # Delete item: del cache[key]
    del cache["What is the speed of light?"]
    print(f"Deleted 'What is the speed of light?'. New len -> {len(cache)}")

    cache.close()


def demo_ttl_expiration() -> None:
    section("4. TIME-TO-LIVE (TTL) & EXPIRATION TIMERS")
    print("Set countdown timers so answers expire automatically.\n")

    cache = TieredSemanticCache()

    # Save with 2-second TTL
    cache.set("Temporary verification code", "948201", ttl=2)
    print("Saved 'Temporary verification code' with 2-second TTL.")

    # Check remaining TTL
    remaining = cache.ttl("Temporary verification code")
    print(f"Remaining TTL: {remaining} seconds")

    res = cache.get("Temporary verification code")
    print(f"Immediate read: {res.value if res else 'None'}")

    print("Waiting 2.5 seconds for timer to expire...")
    time.sleep(2.5)

    # After expiration, it is automatically purged!
    res_expired = cache.get("Temporary verification code")
    print(f"Read after 2.5s: {res_expired} (Cleanly expired and removed!)")

    cache.close()


def demo_tag_invalidation() -> None:
    section("5. GROUP TAG INVALIDATION")
    print("Label answers with tags and delete entire categories in 1 step!\n")

    cache = TieredSemanticCache()

    cache.set("Final score match A", "Team 1 won 3-0", tags=["sports", "football"])
    cache.set("Final score match B", "Team 2 won 2-1", tags=["sports", "football"])
    cache.set("Stock price AAPL", "$185.20", tags=["finance"])

    print(f"Initial total items: {len(cache)}")

    # Invalidate all items tagged with "football"
    deleted_count = cache.invalidate_tag("football")
    print(f"Invalidated tag 'football': removed {deleted_count} items.")
    print(f"Remaining items in cache : {len(cache)} (Only finance remains!)")

    cache.close()


def demo_multi_tenant_isolation() -> None:
    section("6. MULTI-TENANT ISOLATION (Ram vs Shyam)")
    print("Prevent User A from seeing User B's private data, even with identical prompts!\n")

    global_cache = TieredSemanticCache()

    # Create isolated sub-views for each user
    ram_cache = global_cache.namespace("user_ram")
    shyam_cache = global_cache.namespace("user_shyam")

    # Ram saves his private balance
    ram_cache.set("What is my current bank balance?", "$5,000 USD")
    print("User Ram stored his private balance.")

    # Ram asks -> Gets his balance
    ram_result = ram_cache.get("What is my current bank balance?")
    print(f"Ram reads his balance    : {ram_result.value if ram_result else 'None'}")

    # Shyam asks the same question -> Security guard blocks access!
    shyam_result = shyam_cache.get("What is my current bank balance?")
    print(f"Shyam reads same question: {shyam_result} (Cross-tenant access BLOCKED!)")

    global_cache.close()


def demo_stats_and_compaction() -> None:
    section("7. SYSTEM METRICS & DISK COMPACTION")
    print("Inspect cache health metrics and reclaim unused disk space.\n")

    cache = TieredSemanticCache()
    cache.set("query 1", "answer 1")
    cache.set("query 2", "answer 2")

    # Read stats
    stats = cache.stats()
    print("Cache Metrics:")
    for k, v in stats.items():
        print(f"    {k:<20}: {v}")

    # Compact disk file
    reclaimed = cache.compact()
    print(f"\nCompacted disk file: {reclaimed} dead bytes reclaimed.")

    cache.clear()
    print("Cleared cache completely.")
    cache.close()


def demo_web_frameworks() -> None:
    section("8. WEB FRAMEWORKS: FASTAPI & DJANGO INTEGRATION")
    print("Integrating TieredSemanticCache into FastAPI or Django takes just 5 simple lines!\n")

    print("[A] FastAPI Implementation (Ready to Copy):")
    print("    from fastapi import FastAPI")
    print("    from semantic_cache import TieredSemanticCache")
    print("")
    print("    app = FastAPI()")
    print("    cache = TieredSemanticCache()")
    print("")
    print("    # Seed question and answer:")
    print("    cache.set('what is my age', '18')")
    print("")
    print("    @app.get('/ask')")
    print("    def ask(question: str):")
    print("        # Works for BOTH:")
    print("        # 1. Exact match:   'what is my age' -> 18 (instant <1µs)")
    print("        # 2. Semantic match: 'tell me my age' -> 18 (meaning match!)")
    print("        hit = cache.get(question)")
    print("        if hit:")
    print("            return {")
    print("                'answer': hit.value,")
    print("                'matched_key': hit.matched_key,")
    print("                'tier': hit.tier,")
    print("                'similarity': round(hit.similarity, 2),")
    print("            }")
    print("        return {'error': 'Answer not found'}")
    print("")
    print("    # Run with: uvicorn main:app --reload")

    print("\n" + "-" * 75)
    print("[B] Django Implementation (in views.py - Ready to Copy):")
    print("    from django.http import JsonResponse")
    print("    from semantic_cache import TieredSemanticCache")
    print("")
    print("    # Create one shared cache instance")
    print("    cache = TieredSemanticCache()")
    print("    cache.set('what is my age', '18')")
    print("")
    print("    def chat_view(request):")
    print("        question = request.GET.get('q', '')")
    print("        hit = cache.get(question)")
    print("        if hit:")
    print("            return JsonResponse({")
    print("                'answer': hit.value,")
    print("                'matched_key': hit.matched_key,")
    print("                'similarity': round(hit.similarity, 2),")
    print("            })")
    print("        return JsonResponse({'error': 'Not found'}, status=404)")


def main() -> None:
    print("=" * 75)
    print(">>> TIERED SEMANTIC CACHE - FULL FEATURE DEMONSTRATION <<<")
    print("=" * 75)

    demo_standalone_embedder()
    demo_basic_cache()
    demo_dictionary_syntax()
    demo_ttl_expiration()
    demo_tag_invalidation()
    demo_multi_tenant_isolation()
    demo_stats_and_compaction()
    demo_web_frameworks()

    # Clean up test database file if created
    db_file = Path("cache.db")
    if db_file.exists():
        try:
            db_file.unlink()
        except Exception:
            pass

    print("\n" + "=" * 75)
    print("All feature demos completed successfully! Everything is ready to use.")
    print("=" * 75)


if __name__ == "__main__":
    main()

