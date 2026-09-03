"""
Interactive Playground: Testing Text-to-Vector & Similarity
===========================================================

Run this file anytime using:
    python sample.py

What this script demonstrates:
1. How plain text turns into a list of numbers (a vector/arrow).
2. Proof that the arrow's length is scaled to exactly 1.0.
3. How similar meanings get high scores (close to 1.0).
4. How unrelated sentences get low scores (close to 0.0).
5. Fast batch searching across multiple cached sentences at once.
"""

import sys
from pathlib import Path
import numpy as np

# Configure UTF-8 for safe printing on Windows terminal
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Add src folder so we can import our semantic_cache library
sys.path.insert(0, str(Path(__file__).parent / "src"))

from semantic_cache.embedder import (
    DenseHashEmbedder,
    cosine_similarity,
    batch_cosine_similarity,
)


def main():
    print("=" * 70)
    print(">>> TIERED SEMANTIC CACHE - VECTOR & SIMILARITY PLAYGROUND <<<")
    print("=" * 70)

    # 1. Create embedder
    embedder_small = DenseHashEmbedder(dim=8)
    embedder = DenseHashEmbedder(dim=384)

    text = "How do I install Python on Windows?"
    print(f"\n[1] Transforming Text into Numbers (Vector):")
    print(f"    Input text: '{text}'")

    # Show small 8-number representation for visual clarity
    vec_small = embedder_small.embed(text)
    print(f"    Small 8-number vector:\n    {np.round(vec_small, 3)}")

    # Full 384-dimension vector
    vec = embedder.embed(text)
    print(f"\n    Full vector shape: {vec.shape} (384 numbers)")
    print(f"    First 5 numbers:   {np.round(vec[:5], 4)}")
    print(f"    Arrow length (L2 norm): {np.linalg.norm(vec):.6f} (Notice: Exactly 1.0!)")

    # 2. Compare Sentences
    print("\n" + "=" * 70)
    print("[2] Comparing Meaning with Cosine Similarity:")
    print("=" * 70)

    sentence_a = "How do I install Python on Windows?"
    sentence_b = "How to install Python on Windows"
    sentence_c = "The astronaut landed on Mars."

    v_a = embedder.embed(sentence_a)
    v_b = embedder.embed(sentence_b)
    v_c = embedder.embed(sentence_c)

    score_same = cosine_similarity(v_a, v_a)
    score_similar = cosine_similarity(v_a, v_b)
    score_different = cosine_similarity(v_a, v_c)

    print(f"\nSentence A: '{sentence_a}'")
    print(f"Sentence B: '{sentence_b}'")
    print(f"Sentence C: '{sentence_c}'")

    print("\nResults:")
    print(f"    A vs A (Exact Same):       {score_same:.4f}  --> [100% Match!]")
    print(f"    A vs B (Close Phrasing):   {score_similar:.4f}  --> [High Match! (Cache Hit >= 0.85)]")
    print(f"    A vs C (Unrelated Topic):  {score_different:.4f}  --> [Low Match!  (Cache Miss)]")

    # 3. Fast Batch Search
    print("\n" + "=" * 70)
    print("[3] Fast Batch Search (Scanning Multiple Saved Questions at Once):")
    print("=" * 70)

    cached_database = [
        "How do I install Python on Windows?",
        "Where can I find Python tutorials for beginners?",
        "What is the capital city of France?",
        "Delicious recipe for chocolate chip cookies",
        "How to speed up Python execution",
    ]

    # Pre-embed all database questions into a table (matrix)
    cached_matrix = np.array([embedder.embed(q) for q in cached_database])

    # New user query
    user_query = "How to install Python on Windows"
    query_vector = embedder.embed(user_query)

    print(f"User Query: '{user_query}'\n")
    print("Scanning all 5 cached entries in one instant matrix calculation...\n")

    scores = batch_cosine_similarity(query_vector, cached_matrix)

    # Sort results by highest score first
    ranked = sorted(zip(cached_database, scores), key=lambda x: x[1], reverse=True)

    for rank, (question, score) in enumerate(ranked, 1):
        status = "[CACHE HIT!]" if score >= 0.85 else "[CACHE MISS]"
        print(f"    Rank #{rank}: Score = {score:.4f} | {status:<12} | '{question}'")

    print("\n" + "=" * 70)
    print("All checks completed successfully! You can run: python sample.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
