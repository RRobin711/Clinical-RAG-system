from __future__ import annotations

from typing import Any, Dict, List, Sequence

from .retriever import RetrievedChunk


def _format_source(rc: RetrievedChunk, *, max_chars: int = 1200) -> str:
    chunk = rc.chunk
    text = chunk["text"].strip()
    if len(text) > max_chars:
        text = text[: max_chars - 3].rstrip() + "..."
    
    source_id = chunk.get("chunk_id", f"chunk_{rc.rank}")
    source_name = chunk.get("source", "unknown_source")
    
    return (
        f"[{source_id}] (score={rc.score:.4f}, source={source_name})\n"
        f"{text}"
    )


def build_grounded_messages(
    *,
    question: str,
    retrieved: Sequence[RetrievedChunk],
    max_chunk_chars: int = 1200,
) -> List[Dict[str, Any]]:
    """
    Returns OpenAI Chat Completions messages for grounded QA.
    Developer message sets constraints; user message provides question + sources.
    """
    sources = "\n\n".join(_format_source(rc, max_chars=max_chunk_chars) for rc in retrieved)
    
    developer = (
        "You are a careful research assistant. Answer the user's question ONLY using the provided sources. "
        "If the sources do not contain enough information, say you don't have enough evidence.\n\n"
        "Rules:\n"
        "- Do not invent facts.\n"
        "- When you make a claim, cite the supporting source IDs in square brackets, e.g., [chunk_12].\n"
        "- Prefer quoting exact numbers with the surrounding context.\n"
        "- This is not medical advice; present information as literature summary.\n"
    )
    
    user = (
        f"Question:\n{question.strip()}\n\n"
        f"Sources:\n{sources}\n\n"
        "Write a short, structured answer. Include citations like [source_id]."
    )
    
    return [
        {"role": "developer", "content": developer},
        {"role": "user", "content": user},
    ]