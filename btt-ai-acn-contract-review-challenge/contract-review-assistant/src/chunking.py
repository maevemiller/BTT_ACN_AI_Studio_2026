"""
Text chunking strategies for long contract documents.

Two strategies:
    sentence_chunks  — split on sentence boundaries; good for chunk classification
    sliding_window   — overlapping token windows; required for span extraction
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Chunk:
    text: str
    start_char: int
    end_char: int
    chunk_index: int


# ---------------------------------------------------------------------------
# Sentence / paragraph chunking (recommended starting point)
# ---------------------------------------------------------------------------

def sentence_chunks(
    text: str,
    max_chars: int = 512,
) -> list[Chunk]:
    """
    Split contract text into sentence-level chunks no longer than max_chars.

    Adjacent short sentences are merged to approach max_chars without
    exceeding it, balancing context preservation against encoder limits.
    """
    # Naive sentence splitter — replace with spaCy if higher accuracy needed
    raw_sentences = re.split(r"(?<=[.?!])\s+", text)

    chunks: list[Chunk] = []
    current_parts: list[str] = []
    current_start = 0
    cursor = 0

    for sentence in raw_sentences:
        if current_parts and len(" ".join(current_parts)) + len(sentence) + 1 > max_chars:
            chunk_text = " ".join(current_parts)
            chunks.append(Chunk(chunk_text, current_start, current_start + len(chunk_text), len(chunks)))
            current_start = cursor
            current_parts = []

        current_parts.append(sentence)
        cursor += len(sentence) + 1  # +1 for the space consumed by the split

    if current_parts:
        chunk_text = " ".join(current_parts)
        chunks.append(Chunk(chunk_text, current_start, current_start + len(chunk_text), len(chunks)))

    return chunks


# ---------------------------------------------------------------------------
# Sliding-window chunking (stretch goal — span extraction)
# ---------------------------------------------------------------------------

def sliding_window(
    tokens: list[str],
    window_size: int = 384,
    stride: int = 128,
) -> list[list[str]]:
    """
    Produce overlapping windows of tokens for transformer span-prediction.

    Args:
        tokens:      List of tokenized strings (e.g., from a HuggingFace tokenizer).
        window_size: Maximum tokens per window (leave room for special tokens).
        stride:      Step size between window starts. Overlap = window_size - stride.

    Returns:
        List of token windows (each a list of strings).
    """
    windows: list[list[str]] = []
    start = 0
    while start < len(tokens):
        end = min(start + window_size, len(tokens))
        windows.append(tokens[start:end])
        if end == len(tokens):
            break
        start += stride
    return windows
