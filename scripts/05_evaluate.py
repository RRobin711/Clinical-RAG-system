#!/usr/bin/env python3
"""
Day 5: Run retrieval evaluation and log to W&B.

Usage:
    python scripts/05_evaluate.py \\
        --eval_file evaluation/gold_qa.json \\
        --index_dir data/index \\
        --run_name "baseline_fixed2000_naive"
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from clinical_rag.evaluation import (
    EvalSummary,
    load_gold_qa,
    evaluate_retrieval,
    log_to_wandb,
)
from clinical_rag.retriever import Retriever


def print_summary(summary: EvalSummary, results=None) -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  RETRIEVAL EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"  Queries evaluated:  {summary.num_queries}")
    print(f"  Hit Rate @3:        {summary.hit_rate_3:.3f}  ({summary.hit_rate_3*100:.1f}%)")
    print(f"  Hit Rate @5:        {summary.hit_rate_5:.3f}  ({summary.hit_rate_5*100:.1f}%)")
    print(f"  Hit Rate @10:       {summary.hit_rate_10:.3f}  ({summary.hit_rate_10*100:.1f}%)")
    print(f"  MRR:                {summary.mrr:.3f}")
    print(f"{'='*60}")

    # Show failures (most useful for debugging)
    if results:
        failures = [r for r in results if not r.hit_at_5]
        if failures:
            print(f"\n  MISSES (not in top 5):")
            for r in failures:
                print(f"  - Q: {r.question[:80]}...")
                print(f"    Expected: {r.gold_doc_id}")
                top = r.retrieved_doc_ids[:3]
                print(f"    Got: {top}")
                print()


def main():
    parser = argparse.ArgumentParser(description="Day 5: Run retrieval evaluation")
    parser.add_argument("--eval_file", type=Path, default=Path("evaluation/gold_qa.json"))
    parser.add_argument("--index_dir", type=Path, default=Path("data/index"))
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--run_name", type=str, default="eval-baseline")
    parser.add_argument("--wandb_project", type=str, default="clinical-rag")
    parser.add_argument("--no_wandb", action="store_true", help="Skip W&B logging")
    parser.add_argument("--save_results", type=Path, default=None,
                        help="Save detailed results to JSON")
    args = parser.parse_args()

    # Load gold QA
    print(f"Loading gold QA from: {args.eval_file}")
    gold_items = load_gold_qa(args.eval_file)
    print(f"  {len(gold_items)} questions loaded")

    # Load retriever
    print(f"Loading index from: {args.index_dir}")
    retriever = Retriever(index_dir=args.index_dir)
    print(f"  {retriever.index.ntotal} vectors in index")

    # Run evaluation
    print(f"\nRunning evaluation (k={args.k})...")
    results, summary = evaluate_retrieval(retriever, gold_items, k=args.k)

    # Print results
    print_summary(summary, results)

    # Log to W&B
    if not args.no_wandb:
        config = {
            "k": args.k,
            "num_eval_questions": len(gold_items),
            "index_vectors": retriever.index.ntotal,
            "embedding_model": retriever.embedding_model_name,
        }
        log_to_wandb(
            summary,
            project=args.wandb_project,
            run_name=args.run_name,
            config=config,
            results=results,
        )

    # Optionally save detailed results
    if args.save_results:
        args.save_results.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "summary": summary.to_dict(),
            "results": [
                {
                    "question": r.question,
                    "gold_doc_id": r.gold_doc_id,
                    "hit_at_5": r.hit_at_5,
                    "reciprocal_rank": r.reciprocal_rank,
                    "top_5_docs": r.retrieved_doc_ids[:5],
                }
                for r in results
            ],
        }
        args.save_results.write_text(
            json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nDetailed results saved to: {args.save_results}")


if __name__ == "__main__":
    main()
