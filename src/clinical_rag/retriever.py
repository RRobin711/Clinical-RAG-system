from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
from sentence_transformers import SentenceTransformer

from .indexing import load_faiss_index

Metric = Literal["ip", "l2"]


@dataclass(frozen=True)
class RetrievedChunk:
    rank: int
    score: float
    chunk: Dict[str, Any]


class Retriever:
    def __init__(
        self,
        *,
        index_dir: Path,
    ) -> None:
        self.index, self.chunks, self.meta = load_faiss_index(index_dir)
        self.metric: Metric = self.meta.get("metric", "ip")
        self.embedding_model_name: str = self.meta["embedding_model_name"]
        self.normalize_embeddings: bool = bool(self.meta.get("normalize_embeddings", True))
        self.query_instruction: str = str(self.meta.get("query_instruction", "") or "")
        
        self.model = SentenceTransformer(self.embedding_model_name)
        
        if len(self.chunks) != self.index.ntotal:
            raise RuntimeError(
                f"Index has {self.index.ntotal} vectors but chunks.jsonl has {len(self.chunks)} rows. "
                "Rebuild the index."
            )
    
    def _prepare_query(self, query: str) -> str:
        query = (query or "").strip()
        if not query:
            return ""
        if self.query_instruction and not query.startswith(self.query_instruction):
            return self.query_instruction + query
        return query
    
    def retrieve(self, query: str, *, k: int = 5) -> List[RetrievedChunk]:
        query = self._prepare_query(query)
        if not query:
            raise ValueError("Query is empty.")
        
        q_emb = self.model.encode(
            [query],
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        q_emb = np.asarray(q_emb, dtype=np.float32)
        
        scores, indices = self.index.search(q_emb, k)
        scores = scores[0].tolist()
        indices = indices[0].tolist()
        
        results: List[RetrievedChunk] = []
        for i, (idx, score) in enumerate(zip(indices, scores), start=1):
            if idx < 0:
                continue
            results.append(RetrievedChunk(rank=i, score=float(score), chunk=self.chunks[idx]))
        
        return results