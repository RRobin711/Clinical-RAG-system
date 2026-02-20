from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .io_utils import ensure_dir, read_jsonl, write_jsonl

Metric = Literal["ip", "l2"]


@dataclass(frozen=True)
class IndexArtifacts:
    index_path: Path
    chunks_path: Path
    meta_path: Path


def default_bge_query_instruction(model_name: str) -> str:
    """
    For BGE v1.5 models, the authors recommend prefixing short queries with an instruction for retrieval.
    This is a safe, optional improvement for query embeddings only (not corpus embeddings).
    """
    if model_name.strip() == "BAAI/bge-small-en-v1.5":
        return "Represent this sentence for searching relevant passages: "
    return ""


def build_faiss_index(
    *,
    chunks_file: Path,
    index_dir: Path,
    embedding_model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 64,
    normalize_embeddings: bool = True,
    metric: Metric = "ip",
    query_instruction: Optional[str] = None,
    show_progress: bool = True,
) -> IndexArtifacts:
    """
    Builds a FAISS Flat index from a JSONL chunk file.
    - normalize_embeddings=True + metric="ip" approximates cosine similarity.
    - Flat indexes are exact and require no training.
    """
    if not chunks_file.exists():
        raise FileNotFoundError(chunks_file)
    
    ensure_dir(index_dir)
    
    chunks: List[Dict[str, Any]] = list(read_jsonl(chunks_file))
    if not chunks:
        raise RuntimeError(f"No chunks found in {chunks_file}")
    
    texts = [c["text"] for c in chunks]
    model = SentenceTransformer(embedding_model_name)
    
    embeddings_list: List[np.ndarray] = []
    iterator = range(0, len(texts), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Embedding chunks", unit="batch")
    
    for start in iterator:
        batch = texts[start : start + batch_size]
        emb = model.encode(
            batch,
            normalize_embeddings=normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        emb = np.asarray(emb, dtype=np.float32)
        embeddings_list.append(emb)
    
    embeddings = np.concatenate(embeddings_list, axis=0)
    n, d = embeddings.shape
    
    if metric == "ip":
        index = faiss.IndexFlatIP(d)
    elif metric == "l2":
        index = faiss.IndexFlatL2(d)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    index.add(embeddings)
    
    index_path = index_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    
    chunks_path = index_dir / "chunks.jsonl"
    write_jsonl(chunks_path, chunks)
    
    if query_instruction is None:
        query_instruction = default_bge_query_instruction(embedding_model_name)
    
    meta = {
        "embedding_model_name": embedding_model_name,
        "normalize_embeddings": normalize_embeddings,
        "metric": metric,
        "query_instruction": query_instruction,
        "num_chunks": int(n),
        "dimension": int(d),
    }
    
    meta_path = index_dir / "index_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    
    return IndexArtifacts(index_path=index_path, chunks_path=chunks_path, meta_path=meta_path)


def load_faiss_index(index_dir: Path) -> Tuple[faiss.Index, List[Dict[str, Any]], Dict[str, Any]]:
    index_path = index_dir / "faiss.index"
    chunks_path = index_dir / "chunks.jsonl"
    meta_path = index_dir / "index_meta.json"
    
    if not index_path.exists():
        raise FileNotFoundError(index_path)
    if not chunks_path.exists():
        raise FileNotFoundError(chunks_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    
    index = faiss.read_index(str(index_path))
    chunks = list(read_jsonl(chunks_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    
    return index, chunks, meta