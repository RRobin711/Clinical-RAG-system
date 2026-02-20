from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import List, Optional, Set, Tuple

import fitz  # PyMuPDF


def _normalize_whitespace(text: str) -> str:
    # Join hyphenated line breaks: "assess-\nment" -> "assessment"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    # Normalize newlines and spaces
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _page_text_blocks(page: "fitz.Page") -> str:
    """
    Extracts text in a more layout-aware way than plain get_text("text") by using blocks.
    This is still heuristic, but tends to handle multi-column layouts a bit better.
    """
    blocks = page.get_text("blocks")  # (x0, y0, x1, y1, text, block_no, block_type)
    blocks = [b for b in blocks if isinstance(b[4], str) and b[4].strip()]
    blocks.sort(key=lambda b: (round(b[1], 1), round(b[0], 1)))  # sort by y, then x
    return "\n".join(b[4].strip() for b in blocks)


def _detect_repeated_lines(
    pages: List[str],
    *,
    min_pages_ratio: float = 0.6,
    min_line_chars: int = 5,
    max_line_chars: int = 120,
) -> Set[str]:
    """
    Detects likely header/footer lines by counting lines that repeat across many pages.
    """
    if not pages:
        return set()
    
    per_page_lines = []
    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        # Count each line once per page (avoid high counts from repeated table rows)
        per_page_lines.append(set(lines))
    
    counts = Counter()
    for s in per_page_lines:
        counts.update(s)
    
    threshold = max(2, int(len(pages) * min_pages_ratio))
    repeated = set()
    for line, c in counts.items():
        if c >= threshold and min_line_chars <= len(line) <= max_line_chars:
            repeated.add(line)
    
    return repeated


def extract_text_from_pdf(
    pdf_path: Path,
    *,
    max_pages: Optional[int] = None,
    remove_headers_footers: bool = True,
) -> Tuple[str, List[str]]:
    """
    Returns:
        - full_text: concatenated cleaned text for the entire PDF
        - pages: cleaned per-page text list
    """
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    
    pages_raw: List[str] = []
    with fitz.open(str(pdf_path)) as doc:
        n_pages = doc.page_count
        for i in range(n_pages):
            if max_pages is not None and i >= max_pages:
                break
            page = doc.load_page(i)
            page_text = _page_text_blocks(page)
            page_text = _normalize_whitespace(page_text)
            pages_raw.append(page_text)
    
    repeated_lines = _detect_repeated_lines(pages_raw) if remove_headers_footers else set()
    
    pages_clean: List[str] = []
    for p in pages_raw:
        if repeated_lines:
            lines = [ln for ln in p.splitlines() if ln.strip() and ln.strip() not in repeated_lines]
            p = "\n".join(lines).strip()
            p = _normalize_whitespace(p)
        pages_clean.append(p)
    
    full_text = "\n\n".join([p for p in pages_clean if p])
    full_text = _normalize_whitespace(full_text)
    
    return full_text, pages_clean