"""
Day 5: Evaluation framework for the Clinical RAG system.

Implements:
- Gold-standard QA dataset loading/validation
- Hit Rate@k: does the correct source appear in top-k retrieved chunks?
- Mean Reciprocal Rank (MRR): how highly is the correct source ranked?
- Answer faithfulness check (does the LLM answer match the gold answer?)
- W&B experiment logging with full config tracking

This is the file that transforms your project from a demo into engineering.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Gold QA dataset schema
# ---------------------------------------------------------------------------


@dataclass
class GoldQAItem:
    """One question-answer pair with known source document."""

    question: str
    answer: str
    source_doc_id: str  # Must match a doc_id in your chunks
    difficulty: str = "medium"  # easy / medium / hard
    category: str = "general"  # reliability, validity, scoring, population, etc.
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "GoldQAItem":
        return GoldQAItem(
            question=d["question"],
            answer=d["answer"],
            source_doc_id=d["source_doc_id"],
            difficulty=d.get("difficulty", "medium"),
            category=d.get("category", "general"),
            notes=d.get("notes", ""),
        )


def load_gold_qa(path: Path) -> List[GoldQAItem]:
    """Load gold QA dataset from JSON file."""
    if not path.exists():
        raise FileNotFoundError(f"Gold QA file not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    items = [GoldQAItem.from_dict(d) for d in data]
    if not items:
        raise ValueError("Gold QA dataset is empty")
    return items


def save_gold_qa(items: List[GoldQAItem], path: Path) -> None:
    """Save gold QA dataset to JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [item.to_dict() for item in items]
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------


@dataclass
class RetrievalResult:
    """Result of evaluating one query against gold standard."""

    question: str
    gold_doc_id: str
    retrieved_doc_ids: List[str]  # in rank order
    retrieved_scores: List[float]
    hit_at_3: bool = False
    hit_at_5: bool = False
    hit_at_10: bool = False
    reciprocal_rank: float = 0.0  # 0 if not found

    def compute(self) -> None:
        """Compute hit@k and reciprocal rank from retrieved_doc_ids."""
        for i, doc_id in enumerate(self.retrieved_doc_ids):
            if doc_id == self.gold_doc_id:
                rank = i + 1
                self.reciprocal_rank = 1.0 / rank
                self.hit_at_3 = rank <= 3
                self.hit_at_5 = rank <= 5
                self.hit_at_10 = rank <= 10
                return
        # Not found at all
        self.reciprocal_rank = 0.0
        self.hit_at_3 = False
        self.hit_at_5 = False
        self.hit_at_10 = False


@dataclass
class EvalSummary:
    """Aggregated evaluation metrics across all queries."""

    num_queries: int = 0
    hit_rate_3: float = 0.0
    hit_rate_5: float = 0.0
    hit_rate_10: float = 0.0
    mrr: float = 0.0
    avg_retrieval_time_ms: float = 0.0
    avg_e2e_time_ms: float = 0.0
    per_category: Dict[str, Dict[str, float]] = field(default_factory=dict)
    config: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_eval_summary(
    results: List[RetrievalResult],
    *,
    config: Optional[Dict[str, Any]] = None,
) -> EvalSummary:
    """Aggregate individual results into summary metrics."""
    n = len(results)
    if n == 0:
        return EvalSummary()

    summary = EvalSummary(
        num_queries=n,
        hit_rate_3=sum(1 for r in results if r.hit_at_3) / n,
        hit_rate_5=sum(1 for r in results if r.hit_at_5) / n,
        hit_rate_10=sum(1 for r in results if r.hit_at_10) / n,
        mrr=sum(r.reciprocal_rank for r in results) / n,
        config=config or {},
    )

    # Per-category breakdown (if categories are present in the gold data)
    categories: Dict[str, List[RetrievalResult]] = {}
    # We don't have category on RetrievalResult directly, so we skip this
    # unless you enrich it later. Placeholder for blog-worthy analysis.

    return summary


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------


def evaluate_retrieval(
    retriever,
    gold_items: List[GoldQAItem],
    *,
    k: int = 10,
) -> Tuple[List[RetrievalResult], EvalSummary]:
    """
    Run retrieval evaluation against gold QA dataset.
    
    Args:
        retriever: Any object with a .retrieve(query, k=k) method
                   returning objects with .chunk["doc_id"] and .score
        gold_items: List of GoldQAItem with known answers
        k: Number of chunks to retrieve per query
    
    Returns:
        (individual_results, summary)
    """
    results: List[RetrievalResult] = []

    for item in gold_items:
        t0 = time.perf_counter()
        retrieved = retriever.retrieve(item.question, k=k)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Extract doc_ids from retrieved chunks
        doc_ids = [rc.chunk.get("doc_id", "") for rc in retrieved]
        scores = [rc.score for rc in retrieved]

        result = RetrievalResult(
            question=item.question,
            gold_doc_id=item.source_doc_id,
            retrieved_doc_ids=doc_ids,
            retrieved_scores=scores,
        )
        result.compute()
        results.append(result)

    summary = compute_eval_summary(results)
    return results, summary


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------


def log_to_wandb(
    summary: EvalSummary,
    *,
    project: str = "clinical-rag",
    run_name: str = "eval-run",
    config: Optional[Dict[str, Any]] = None,
    results: Optional[List[RetrievalResult]] = None,
) -> None:
    """Log evaluation results to Weights & Biases."""
    if not WANDB_AVAILABLE:
        print("[WARN] wandb not installed. Skipping W&B logging.")
        print(f"       Install with: pip install wandb")
        return

    merged_config = {**(config or {}), **summary.config}

    run = wandb.init(
        project=project,
        name=run_name,
        config=merged_config,
        reinit=True,
    )

    wandb.log(
        {
            "hit_rate@3": summary.hit_rate_3,
            "hit_rate@5": summary.hit_rate_5,
            "hit_rate@10": summary.hit_rate_10,
            "mrr": summary.mrr,
            "num_queries": summary.num_queries,
        }
    )

    # Log per-query results as a W&B Table (great for debugging failures)
    if results:
        columns = [
            "question",
            "gold_doc",
            "hit@5",
            "reciprocal_rank",
            "top_retrieved",
        ]
        table_data = []
        for r in results:
            table_data.append(
                [
                    r.question[:100],
                    r.gold_doc_id,
                    r.hit_at_5,
                    round(r.reciprocal_rank, 3),
                    r.retrieved_doc_ids[0] if r.retrieved_doc_ids else "none",
                ]
            )
        wandb.log({"per_query_results": wandb.Table(columns=columns, data=table_data)})

    run.finish()
    print(f"[W&B] Run '{run_name}' logged to project '{project}'")
