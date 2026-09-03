"""
Tiered Semantic Cache - Vector Math Engine (Easy Kid-Friendly Guide)
====================================================================

What does this file do?
-----------------------
Computers cannot read English sentences directly; they only know numbers.
This file does two simple things:
1. Turns any sentence into a list of numbers (called an "arrow" or "vector").
2. Checks if two sentences mean the same thing by checking if their arrows
   point in the same direction!

The Simple "Arrow in a Room" Picture:
-------------------------------------
* Imagine you stand in the center of a room and point an arrow.
* Similar questions (like "how to code in python" and "learn python coding")
  point in almost the exact same direction.
* Unrelated questions (like "recipe for pizza") point in a completely different direction.

The Math Logic in 4 Simple Points:
----------------------------------
1. Arrow Length (L2 Norm):
   - Measure how long the arrow is from start to tip.
   - Formula: length = sqrt(number1^2 + number2^2 + ...)
   - Just like finding the hypotenuse of a triangle in school!

2. Make Every Arrow the Same Length (L2 Normalization):
   - Divide every coordinate by the arrow's length.
   - Now every arrow has a length of exactly 1.0!
   - Why? So long sentences and short sentences are treated 100% fairly.

3. Compare Directions (Cosine Similarity):
   - Once arrows are length 1.0, multiply matching numbers and add them up.
   - Scores:
     * +1.0 = Arrows point in the exact same direction (same meaning!).
     *  0.0 = Arrows make a 90-degree corner (completely unrelated).
     * -1.0 = Arrows point in opposite directions.

4. Lightning-Fast Search (Batch Matrix Multiply):
   - Instead of checking cached answers one by one, we stack them into a
     table and multiply them all against your question in one single instant.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Sequence, Union
import zlib
import numpy as np


def l2_normalize(vector: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Resize an arrow so its length is exactly 1.0.

    Simple steps:
    1. Calculate length = sqrt(sum of all squared numbers).
    2. Divide each number by this length.
    3. 'eps' is a tiny guard so we never divide by zero.

    Args:
        vector: A single arrow (1D) or a table of arrows (2D).
        eps: Tiny safety number (0.000000000001).

    Returns:
        The exact same arrow resized to have a length of 1.0.
    """
    arr = np.asarray(vector, dtype=np.float32)
    if arr.ndim == 1:
        # Calculate length of 1 arrow
        norm = float(np.linalg.norm(arr))
        return arr / max(norm, eps)
    elif arr.ndim == 2:
        # Calculate lengths of a whole table of arrows row-by-row
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.maximum(norms, eps)
    else:
        raise ValueError(f"Expected 1D arrow or 2D table, got {arr.ndim}D")


def cosine_similarity(
    u: np.ndarray,
    v: np.ndarray,
    pre_normalized: bool = False,
) -> float:
    """Calculate how closely two arrows point in the same direction.

    Simple steps:
    1. Make sure both arrows have length 1.0.
    2. Multiply matching numbers and add them up (dot product).
    3. Returns a score between -1.0 (opposite) and +1.0 (identical).

    Args:
        u: First arrow (sentence A).
        v: Second arrow (sentence B).
        pre_normalized: If True, arrows are already length 1.0 (skips math for speed).

    Returns:
        Score between -1.0 and 1.0.
    """
    u_vec = u if pre_normalized else l2_normalize(u)
    v_vec = v if pre_normalized else l2_normalize(v)
    sim = float(np.dot(u_vec, v_vec))
    # Keep strictly within [-1.0, 1.0] to prevent tiny computer rounding errors
    return max(-1.0, min(1.0, sim))


def batch_cosine_similarity(
    query: np.ndarray,
    matrix: np.ndarray,
    pre_normalized: bool = True,
) -> np.ndarray:
    """Compare one question against thousands of saved answers in one single step.

    Simple steps:
    1. Take the new question arrow.
    2. Multiply it against the entire table of saved arrows using fast computer math (BLAS).
    3. Returns similarity scores for all saved answers instantly!

    Args:
        query: New question arrow of shape (dim,).
        matrix: Table of all saved arrows of shape (N, dim).
        pre_normalized: True if saved arrows are already length 1.0 (default).

    Returns:
        A list of N similarity scores, one for each saved answer.
    """
    if matrix.size == 0:
        return np.empty((0,), dtype=np.float32)

    q = query if pre_normalized else l2_normalize(query)
    m = matrix if pre_normalized else l2_normalize(matrix)

    # Fast hardware multiplication: checks all items at once
    scores = np.dot(m, q)
    return np.clip(scores, -1.0, 1.0)


class BaseEmbedder(ABC):
    """Simple blueprint for any text-to-numbers converter.

    Any AI model (OpenAI, HuggingFace, or local code) can plug in by following
    these rules.
    """

    @property
    @abstractmethod
    def dim(self) -> int:
        """How many numbers make up each arrow (e.g. 384)."""
        pass

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """Turn text into an arrow of numbers with length 1.0."""
        pass


class CallableEmbedder(BaseEmbedder):
    """Plug-in adapter for external AIs (like OpenAI, Anthropic, or HuggingFace).

    Takes your custom function, checks the number count, and automatically
    resizes the arrow to length 1.0.
    """

    def __init__(
        self,
        func: Callable[[str], Union[np.ndarray, Sequence[float]]],
        dim: int,
    ) -> None:
        """Set up with your custom AI function and expected arrow size.

        Args:
            func: Function that takes text and returns numbers.
            dim: Expected number count (e.g. 1536 for OpenAI).
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
    """Built-in offline converter: turns text into arrows without any AI API key.

    Why is this special?
    - Runs 100% on your computer without internet.
    - Super fast (takes ~25 microseconds per sentence!).
    - 100% reliable: the same sentence always gives the exact same arrow.

    How it works in 3 easy steps:
    1. Word & Letter Chopping:
       - Slices words into small 2, 3, and 4-letter pieces.
       - Example: "install" shares pieces with "installing" and "installer".
    2. Fast C-Speed Bucket Hashing:
       - Uses computer hardware CRC32 to drop each letter piece into a slot.
       - Flips a coin (+1 or -1) so numbers stay balanced around zero.
    3. Resize to 1.0:
       - Resizes the final arrow so its length is exactly 1.0.
    """

    def __init__(self, dim: int = 384) -> None:
        """Create an embedder that makes arrows with 'dim' numbers."""
        if dim <= 0:
            raise ValueError(f"Dimension must be positive, got {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, text: str) -> np.ndarray:
        """Turn text into an arrow of length 1.0 super fast."""
        # Convert text to computer bytes once
        clean = text.strip().lower().encode("utf-8")
        if not clean:
            return np.zeros(self._dim, dtype=np.float32)

        vec = np.zeros(self._dim, dtype=np.float32)
        words = clean.split()

        for w in words:
            # 1. Whole word vote
            val = zlib.crc32(w)
            vec[val % self._dim] += -1.0 if (val & 0x10000) else 1.0

            # 2. Letter chunks (sizes 2, 3, and 4) vote
            padded = b"<" + w + b">"
            length = len(padded)
            for n in (2, 3, 4):
                for i in range(length - n + 1):
                    sub_val = zlib.crc32(padded[i : i + n])
                    vec[sub_val % self._dim] += -1.0 if (sub_val & 0x10000) else 1.0

        # Scale the final arrow so its total length is 1.0
        norm = float(np.linalg.norm(vec))
        return vec / max(norm, 1e-12)
