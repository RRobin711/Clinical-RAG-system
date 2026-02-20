from __future__ import annotations

import re
from typing import List


def estimate_tokens(text: str) -> int:
    """
    Very rough token estimate if you don't want to pull in a tokenizer dependency.
    Rule of thumb: ~4 chars per token for English.
    """
    text = text.strip()
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def _clean_for_chunking(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(
    text: str,
    *,
    chunk_size_chars: int = 2000,
    overlap_chars: int = 200,
    min_chunk_chars: int = 300,
) -> List[str]:
    """
    Naive fixed-size chunking with overlap (Day 2 baseline).
    - Uses characters (not tokens) to keep dependencies minimal.
    - Attempts to break on whitespace so chunks don't end in the middle of a word.
    """
    if chunk_size_chars <= 0:
        raise ValueError("chunk_size_chars must be > 0")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= chunk_size_chars:
        raise ValueError("overlap_chars must be smaller than chunk_size_chars")
    
    text = _clean_for_chunking(text)
    if not text:
        return []
    
    chunks: List[str] = []
    start = 0
    n = len(text)
    
    while start < n:
        end = min(start + chunk_size_chars, n)
        
        # Extend end forward to the next whitespace to avoid cutting a word
        if end < n:
            while end < n and not text[end].isspace():
                end += 1
            if end - start > chunk_size_chars + 200:
                break
        
        chunk = text[start:end].strip()
        if len(chunk) >= min_chunk_chars:
            chunks.append(chunk)
        
        if end >= n:
            break
        
        start = max(0, end - overlap_chars)
    
    return chunks