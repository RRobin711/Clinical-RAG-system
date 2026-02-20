"""
Day 6: Chunking strategies and HyDE (Hypothetical Document Embeddings).

Implements:
- Fixed 512-char chunking (your Day 2 baseline)
- Fixed 1024-char chunking (larger context window)
- Semantic chunking (split on paragraph/section boundaries)
- HyDE retriever wrapper (Gao et al., 2022)

The 2x2 experiment matrix: (Naive vs HyDE) x (Best Chunking vs 512 Baseline)
is the single most impressive thing in this project.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import chunk_text


# ---------------------------------------------------------------------------
# Chunking strategies
# ---------------------------------------------------------------------------


def chunk_fixed_512(text: str) -> List[str]:
    """Baseline: 512-char chunks with 50-char overlap (your Day 2 strategy)."""
    return chunk_text(text, chunk_size_chars=512, overlap_chars=50, min_chunk_chars=100)


def chunk_fixed_1024(text: str) -> List[str]:
    """Larger chunks: 1024-char with 100-char overlap. More context per chunk."""
    return chunk_text(text, chunk_size_chars=1024, overlap_chars=100, min_chunk_chars=200)


def chunk_fixed_2000(text: str) -> List[str]:
    """Even larger: 2000-char with 200-char overlap (the Day 0-4 starter default)."""
    return chunk_text(text, chunk_size_chars=2000, overlap_chars=200, min_chunk_chars=300)


# Section header patterns common in clinical psychology papers
_SECTION_PATTERNS = [
    r"^(?:Abstract|ABSTRACT)\s*$",
    r"^(?:Introduction|INTRODUCTION)\s*$",
    r"^(?:Background|BACKGROUND)\s*$",
    r"^(?:Methods?|METHODS?)\s*$",
    r"^(?:Materials?\s+and\s+Methods?)\s*$",
    r"^(?:Results?|RESULTS?)\s*$",
    r"^(?:Discussion|DISCUSSION)\s*$",
    r"^(?:Conclusions?|CONCLUSIONS?)\s*$",
    r"^(?:References?|REFERENCES?)\s*$",
    r"^(?:Acknowledgements?|ACKNOWLEDGEMENTS?)\s*$",
    r"^(?:Supplementary|SUPPLEMENTARY)\s",
    r"^(?:Statistical\s+Analysis|Data\s+Analysis)\s*$",
    r"^(?:Participants?|Subjects?|Sample)\s*$",
    r"^(?:Measures?|Instruments?|Scales?)\s*$",
    r"^(?:Procedure)\s*$",
    r"^\d+\.?\s+[A-Z]",  # Numbered sections like "1. Introduction"
]

_SECTION_RE = re.compile("|".join(_SECTION_PATTERNS), re.MULTILINE | re.IGNORECASE)


def chunk_semantic(
    text: str,
    *,
    max_chunk_chars: int = 2000,
    min_chunk_chars: int = 200,
) -> List[str]:
    """
    Semantic chunking: split on paragraph boundaries and section headers.
    
    Strategy:
    1. Split text on double-newlines (paragraph boundaries)
    2. Further split on detected section headers
    3. Merge small consecutive paragraphs up to max_chunk_chars
    4. Never split mid-paragraph unless paragraph exceeds max_chunk_chars
    
    This preserves logical structure from academic papers.
    """
    if not text.strip():
        return []

    # Step 1: Split on double newlines (paragraph boundaries)
    raw_paragraphs = re.split(r"\n\s*\n", text)
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]

    # Step 2: Further split at section headers
    segments: List[str] = []
    for para in raw_paragraphs:
        # Check if this paragraph contains a section header
        lines = para.split("\n")
        current_segment = []
        for line in lines:
            if _SECTION_RE.match(line.strip()) and current_segment:
                # Flush accumulated lines as one segment
                segments.append("\n".join(current_segment).strip())
                current_segment = [line]
            else:
                current_segment.append(line)
        if current_segment:
            segments.append("\n".join(current_segment).strip())

    segments = [s for s in segments if s]

    # Step 3: Merge small consecutive segments, respect max_chunk_chars
    chunks: List[str] = []
    buffer = ""

    for seg in segments:
        candidate = (buffer + "\n\n" + seg).strip() if buffer else seg

        if len(candidate) <= max_chunk_chars:
            buffer = candidate
        else:
            # Flush buffer if it has content
            if buffer and len(buffer) >= min_chunk_chars:
                chunks.append(buffer)
            elif buffer:
                # Buffer too small on its own, force-merge with next
                buffer = candidate
                continue

            # Handle oversized segment: fall back to fixed chunking
            if len(seg) > max_chunk_chars:
                sub_chunks = chunk_text(
                    seg,
                    chunk_size_chars=max_chunk_chars,
                    overlap_chars=100,
                    min_chunk_chars=min_chunk_chars,
                )
                chunks.extend(sub_chunks)
                buffer = ""
            else:
                buffer = seg

    # Flush remaining buffer
    if buffer and len(buffer) >= min_chunk_chars:
        chunks.append(buffer)
    elif buffer and chunks:
        # Merge tiny leftover into last chunk
        chunks[-1] = chunks[-1] + "\n\n" + buffer

    return chunks


# ---------------------------------------------------------------------------
# Rechunking pipeline (rebuild chunks.jsonl with a different strategy)
# ---------------------------------------------------------------------------

CHUNKING_STRATEGIES = {
    "fixed_512": chunk_fixed_512,
    "fixed_1024": chunk_fixed_1024,
    "fixed_2000": chunk_fixed_2000,
    "semantic": chunk_semantic,
}


def rechunk_corpus(
    extracted_dir: Path,
    output_file: Path,
    strategy_name: str,
    manifest_path: Optional[Path] = None,
) -> int:
    """
    Re-chunk the entire corpus with a different strategy.
    Reads .txt files from extracted_dir, writes new chunks.jsonl.
    
    Returns: number of chunks produced.
    """
    if strategy_name not in CHUNKING_STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Choose from: {list(CHUNKING_STRATEGIES.keys())}"
        )

    chunk_fn = CHUNKING_STRATEGIES[strategy_name]

    # Load manifest for source PDF mapping (optional)
    manifest_map: Dict[str, str] = {}
    if manifest_path and manifest_path.exists():
        for line in manifest_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") == "ok" and row.get("doc_id"):
                manifest_map[row["doc_id"]] = row.get("source_pdf", f"{row['doc_id']}.pdf")

    txt_files = sorted(extracted_dir.glob("*.txt"))
    if not txt_files:
        raise RuntimeError(f"No .txt files in {extracted_dir}")

    all_chunks: List[Dict[str, Any]] = []

    for txt_path in txt_files:
        doc_id = txt_path.stem
        text = txt_path.read_text(encoding="utf-8")
        chunks = chunk_fn(text)
        source_pdf = manifest_map.get(doc_id, f"{doc_id}.pdf")

        for i, ch in enumerate(chunks):
            all_chunks.append(
                {
                    "chunk_id": f"{doc_id}_{i:05d}",
                    "doc_id": doc_id,
                    "source": source_pdf,
                    "chunk_index": i,
                    "num_chars": len(ch),
                    "text": ch,
                }
            )

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        for chunk in all_chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    return len(all_chunks)


# ---------------------------------------------------------------------------
# HyDE Retriever (Gao et al., 2022)
# ---------------------------------------------------------------------------


class HyDERetriever:
    """
    Hypothetical Document Embeddings retriever.
    
    Paper: Gao et al., 2022 - "Precise Zero-Shot Dense Retrieval 
           without Relevance Labels" (arxiv.org/abs/2212.10496)
    
    Instead of embedding the user's question directly:
    1. Ask the LLM to generate a hypothetical expert answer
    2. Embed that hypothetical answer
    3. Search FAISS with the hypothetical answer embedding
    4. Return the real chunks found
    
    The intuition: the hypothetical answer is linguistically closer
    to the actual answer chunks than the question is, so retrieval
    improves. The hypothetical is ONLY used for retrieval — the final
    answer is still grounded in real sources.
    """

    HYPOTHESIS_PROMPT = (
        "You are an expert in clinical psychology assessment scales. "
        "Write a short, factual paragraph (3-5 sentences) that would directly "
        "answer the following question. Write as if you are stating facts from "
        "a research paper. Do not hedge or say you don't know.\n\n"
        "Question: {question}\n\n"
        "Expert answer:"
    )

    def __init__(
        self,
        *,
        base_retriever,  # Your existing Retriever from retriever.py
        llm_fn,  # callable(prompt: str) -> str
    ):
        self.base_retriever = base_retriever
        self.llm_fn = llm_fn
        self._last_hypothesis: Optional[str] = None

    @property
    def last_hypothesis(self) -> Optional[str]:
        """Access the last generated hypothesis (useful for debugging/logging)."""
        return self._last_hypothesis

    def retrieve(self, query: str, *, k: int = 5):
        """
        HyDE retrieval: generate hypothesis -> embed hypothesis -> search.
        
        Returns same format as base_retriever.retrieve() for compatibility.
        """
        # Step 1: Generate hypothetical answer
        prompt = self.HYPOTHESIS_PROMPT.format(question=query)
        hypothesis = self.llm_fn(prompt)
        self._last_hypothesis = hypothesis

        # Step 2: Embed the hypothesis (not the original question)
        # We use the base retriever's model and settings for consistency
        q_text = hypothesis.strip()
        if self.base_retriever.query_instruction:
            q_text = self.base_retriever.query_instruction + q_text

        q_emb = self.base_retriever.model.encode(
            [q_text],
            normalize_embeddings=self.base_retriever.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        q_emb = np.asarray(q_emb, dtype=np.float32)

        # Step 3: Search FAISS with hypothesis embedding
        scores, indices = self.base_retriever.index.search(q_emb, k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()

        # Step 4: Build results (same format as base retriever)
        from .retriever import RetrievedChunk

        results = []
        for i, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0:
                continue
            results.append(
                RetrievedChunk(
                    rank=i,
                    score=float(score),
                    chunk=self.base_retriever.chunks[idx],
                )
            )

        return results


# ---------------------------------------------------------------------------
# LLM helper for HyDE (works with both Anthropic and OpenAI)
# ---------------------------------------------------------------------------


def make_anthropic_hyde_fn(model: str = "claude-sonnet-4-20250514"):
    """
    Returns a callable that sends a prompt to Anthropic's API.
    Uses ANTHROPIC_API_KEY from environment.
    """
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    def call_llm(prompt: str) -> str:
        response = client.messages.create(
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()

    return call_llm


def make_openai_hyde_fn(model: str = "gpt-4o-mini"):
    """
    Returns a callable that sends a prompt to OpenAI's API.
    Uses OPENAI_API_KEY from environment.
    """
    from openai import OpenAI

    client = OpenAI()

    def call_llm(prompt: str) -> str:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()

    return call_llm
