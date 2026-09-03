"""
Tiered Semantic Cache - Vector Math & Embeddings
===============================================

What is this file?
------------------
Computers do not understand words; they understand numbers.
This file turns text into a list of numbers (called an "embedding" or "vector")
and calculates how similar two texts are based on their meaning.

Intuitive Picture (The "Arrow" Analogy):
----------------------------------------
* Imagine every sentence is an arrow drawn from the center of a room.
* Sentences with similar meanings (like "cute puppy" and "adorable dog")
  point in almost the exact same direction.
* Unrelated sentences (like "cute puppy" and "quantum physics") point in
  completely different directions.

The Math Logic in 4 Simple Points:
----------------------------------
1. Arrow Length (L2 Norm):
   - Just like finding the hypotenuse in school: square each number, add them up,
     and take the square root.
   - Formula: length = sqrt(x^2 + y^2 + z^2 + ...)

2. Make Every Arrow the Same Length (L2 Normalization):
   - Divide the arrow coordinates by its length.
   - Result: Every arrow now has a length of exactly 1.0.
   - Why? It makes comparing directions completely fair, regardless of how long
     the sentence was.

3. Compare Directions (Cosine Similarity):
   - Multiply matching numbers together and add them up (the "dot product").
   - Results:
     * +1.0 = Arrows point in the exact same direction (same meaning).
     *  0.0 = Arrows are at a right angle / 90 degrees (unrelated topics).
     * -1.0 = Arrows point in opposite directions.

4. Super-Fast Search (Batch Matrix Product):
   - Instead of checking cached items one by one, we stack all saved arrows into
     a table (matrix) and multiply them by the new question in one instant step.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import hashlib
from typing import Callable, Sequence, Union
import numpy as np


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Resize an arrow so its length is exactly 1.0.

    Simple steps:
    1. Calculate length = sqrt(sum of all numbers squared).
    2. Divide each number by this length.
    3. 'eps' is a tiny safety number so we never accidentally divide by zero.

    Args:
        vector: A list or table of numbers.
        eps: Tiny guard number (default 0.000000000001).

    Returns:
        The same arrow scaled to have a total length of 1.0.
    """
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim == 1:
        # Calculate length of 1 arrow
        norm = float(np.linalg.norm(arr))
        return arr / max(norm, eps)
    elif arr.ndim == 2:
        # Calculate lengths of multiple arrows row-by-row
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.maximum(norms, eps)
        return arr / norms
    else:
        raise ValueError(f"Expected 1D or 2D list of numbers, got {arr.ndim}D")


def cosine_similarity(u: np.ndarray, v: np.ndarray) -> float:
    """Calculate how closely two arrows point in the same direction.

    Simple steps:
    1. Make both arrows length 1.0.
    2. Multiply matching numbers and sum them up (dot product).
    3. Score is between -1.0 (opposite) and +1.0 (identical).

    Args:
        u: First list of numbers (sentence A).
        v: Second list of numbers (sentence B).

    Returns:
        Similarity score from -1.0 to 1.0.
    """
    u_norm = l2_normalize(u)
    v_norm = l2_normalize(v)
    sim = float(np.dot(u_norm, v_norm))
    # Keep strictly within [-1.0, 1.0] to stop tiny computer rounding glitches
    return max(-1.0, min(1.0, sim))


def batch_cosine_similarity(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compare one question against thousands of cached answers in one single step.

    Simple steps:
    1. Normalize query arrow and all saved arrows to length 1.0.
    2. Do one fast matrix-vector multiplication (BLAS GEMV).
    3. Returns a similarity score for each saved item instantly.

    Args:
        query: The new search query arrow of shape (dim,).
        matrix: A table of all saved cache arrows of shape (N, dim).

    Returns:
        List of N similarity scores, one for each saved item.
    """
    if matrix.size == 0:
        return np.empty((0,), dtype=np.float32)

    q_norm = l2_normalize(query)
    m_norm = l2_normalize(matrix)
    scores = np.dot(m_norm, q_norm)
    return np.clip(scores, -1.0, 1.0)


class BaseEmbedder(ABC):
    """Blueprint for any text-to-numbers converter.
    
    Any AI model (OpenAI, HuggingFace, local code) can plug in by following
    this simple blueprint.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """How many numbers are in each arrow (e.g., 384)."""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Turn text into a list of numbers of length 1.0."""
        pass


class CallableEmbedder(BaseEmbedder):
    """Adapter to let you plug in any AI service (OpenAI, Anthropic, local model).
    
    Takes your custom function, checks that the number count is correct,
    and automatically scales the arrow to length 1.0.
    """

    def __init__(
        self,
        func: Callable[[str], Union[np.ndarray, Sequence[float]]],
        dim: int,
    ) -> None:
        """Set up with your custom function and expected arrow size.

        Args:
            func: A function that takes a text string and returns numbers.
            dim: The number of dimensions expected (e.g., 1536 for OpenAI).
        """
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self._func = func
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        raw_vec = self._func(text)
        arr = np.asarray(raw_vec, dtype=np.float32)
        if arr.shape != (self._dim,):
            raise ValueError(
                f"Embedder returned {arr.shape[0]} numbers, expected {self._dim}"
            )
        return l2_normalize(arr)


class DenseHashEmbedder(BaseEmbedder):
    """Built-in, offline, zero-dependency text-to-numbers converter.
    
    Why use this?
    - Works out of the box with zero internet connection and zero API keys.
    - Fast and 100% deterministic (the same sentence always gives the exact same arrow).

    Simple Math & Logic in Points:
    ------------------------------
    1. Word & Letter Chopping (N-grams):
       - Chops words into little 2, 3, and 4-letter pieces.
       - Example: "search" and "searching" share many identical pieces ("sea", "ear", "arch").
       
    2. Bucket Hashing:
       - Uses a cryptographic hash to map each letter piece to one of 'dim' buckets.
       - Uses a coin-flip sign (+1 or -1) so the average score is zero (unbiased).
       
    3. Scale to 1.0:
       - Scales the final result so the arrow has a length of 1.0.
    """

    def __init__(self, dim: int = 384) -> None:
        """Create an embedder that outputs arrows with 'dim' numbers."""
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def _tokenize(self, text: str) -> list[str]:
        """Cut text into clean lowercase words and small letter chunks."""
        clean = text.strip().lower()
        if not clean:
            return ["<empty>"]

        words = clean.split()
        tokens: list[str] = list(words)

        # Slice each word into pieces of size 2, 3, and 4 letters
        for word in words:
            padded = f"<{word}>"
            length = len(padded)
            for n in (2, 3, 4):
                if length >= n:
                    tokens.extend(padded[i : i + n] for i in range(length - n + 1))

        return tokens

    def embed(self, text: str) -> np.ndarray:
        """Convert input text into an arrow of length 1.0."""
        tokens = self._tokenize(text)
        vec = np.zeros(self._dim, dtype=np.float32)

        # Place each letter chunk into a bucket with a +1 or -1 vote
        for token in tokens:
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            val = int.from_bytes(digest, "little")
            bucket = val % self._dim
            sign = -1.0 if (val >> 32) & 1 else 1.0
            vec[bucket] += sign

        # Finish by scaling the arrow to length 1.0
        return l2_normalize(vec)
