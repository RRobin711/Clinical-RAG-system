#!/usr/bin/env python3
"""
Day 6: Run the full experiment matrix and produce comparison table.

This script is what separates your project from every other RAG tutorial.

Experiment matrix:
  Retrieval: Naive vs HyDE
  Chunking:  fixed_512 vs fixed_1024 vs fixed_2000 vs semantic

Usage:
    # Run all experiments (rechunk + rebuild index + evaluate for each config)
    python scripts/06_run_experiments.py --all

    # Run a single experiment
    python scripts/06_run_experiments.py --chunking semantic --retrieval naive

    # Just print comparison table from saved results
    python scripts/06_run_experiments.py --compare
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from clinical_rag.chunking_strategies import (
    CHUNKING_STRATEGIES,
    HyDERetriever,
    rechunk_corpus,
)
from clinical_rag.evaluation import (
    evaluate_retrieval,
    load_gold_qa,
    log_to_wandb,
)
from clinical_rag.indexing import build_faiss_index
from clinical_rag.retriever import Retriever


RESULTS_DIR = Path("evaluation/experiment_results")
EXTRACTED_DIR = Path("data/extracted")
MANIFEST_PATH = Path("data/extraction_manifest.jsonl")


def run_single_experiment(
    *,
    chunking: str,
    retrieval: str,  # "naive" or "hyde"
    eval_file: Path,
    wandb_project: str = "clinical-rag",
    no_wandb: bool = False,
    llm_provider: str = "anthropic",  # "anthropic" or "openai"
) -> Dict[str, Any]:
    """
    Run one experiment configuration:
    1. Rechunk corpus with the given strategy
    2. Rebuild FAISS index
    3. Evaluate retrieval
    4. Log to W&B
    5. Save results
    """
    exp_name = f"{chunking}_{retrieval}"
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT: {exp_name}")
    print(f"  Chunking: {chunking}  |  Retrieval: {retrieval}")
    print(f"{'='*60}")

    # Paths for this experiment
    exp_chunks = Path(f"data/experiments/{exp_name}/chunks.jsonl")
    exp_index_dir = Path(f"data/experiments/{exp_name}/index")

    # Step 1: Rechunk
    print(f"\n[1/4] Rechunking with strategy='{chunking}'...")
    t0 = time.perf_counter()
    num_chunks = rechunk_corpus(
        extracted_dir=EXTRACTED_DIR,
        output_file=exp_chunks,
        strategy_name=chunking,
        manifest_path=MANIFEST_PATH if MANIFEST_PATH.exists() else None,
    )
    rechunk_time = time.perf_counter() - t0
    print(f"      → {num_chunks} chunks in {rechunk_time:.1f}s")

    # Step 2: Rebuild index
    print(f"\n[2/4] Building FAISS index...")
    t0 = time.perf_counter()
    build_faiss_index(
        chunks_file=exp_chunks,
        index_dir=exp_index_dir,
        embedding_model_name="BAAI/bge-small-en-v1.5",
        normalize_embeddings=True,
        metric="ip",
        show_progress=True,
    )
    index_time = time.perf_counter() - t0
    print(f"      → Index built in {index_time:.1f}s")

    # Step 3: Load retriever
    base_retriever = Retriever(index_dir=exp_index_dir)

    if retrieval == "hyde":
        print(f"\n[3/4] Running HyDE evaluation (provider={llm_provider})...")
        # Create LLM function for HyDE
        if llm_provider == "anthropic":
            from clinical_rag.chunking_strategies import make_anthropic_hyde_fn
            llm_fn = make_anthropic_hyde_fn()
        else:
            from clinical_rag.chunking_strategies import make_openai_hyde_fn
            llm_fn = make_openai_hyde_fn()

        retriever = HyDERetriever(base_retriever=base_retriever, llm_fn=llm_fn)
    else:
        print(f"\n[3/4] Running naive evaluation...")
        retriever = base_retriever

    # Step 4: Evaluate
    gold_items = load_gold_qa(eval_file)
    t0 = time.perf_counter()
    results, summary = evaluate_retrieval(retriever, gold_items, k=10)
    eval_time = time.perf_counter() - t0

    # Print results
    print(f"\n      Hit@3:  {summary.hit_rate_3:.3f}")
    print(f"      Hit@5:  {summary.hit_rate_5:.3f}")
    print(f"      Hit@10: {summary.hit_rate_10:.3f}")
    print(f"      MRR:    {summary.mrr:.3f}")
    print(f"      Time:   {eval_time:.1f}s for {len(gold_items)} queries")

    # Log to W&B
    config = {
        "chunking_strategy": chunking,
        "retrieval_method": retrieval,
        "num_chunks": num_chunks,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "llm_provider": llm_provider if retrieval == "hyde" else "none",
    }

    if not no_wandb:
        log_to_wandb(
            summary,
            project=wandb_project,
            run_name=exp_name,
            config=config,
            results=results,
        )

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_data = {
        "experiment": exp_name,
        "chunking": chunking,
        "retrieval": retrieval,
        "num_chunks": num_chunks,
        "hit_rate_3": summary.hit_rate_3,
        "hit_rate_5": summary.hit_rate_5,
        "hit_rate_10": summary.hit_rate_10,
        "mrr": summary.mrr,
        "eval_time_s": eval_time,
        "num_queries": len(gold_items),
    }
    result_path = RESULTS_DIR / f"{exp_name}.json"
    result_path.write_text(json.dumps(result_data, indent=2), encoding="utf-8")
    print(f"\n[4/4] Results saved to: {result_path}")

    return result_data


def run_all_experiments(
    eval_file: Path,
    no_wandb: bool = False,
    llm_provider: str = "anthropic",
) -> List[Dict[str, Any]]:
    """Run the full experiment matrix."""
    # Chunking strategies to test
    chunking_strategies = ["fixed_512", "fixed_1024", "fixed_2000", "semantic"]
    retrieval_methods = ["naive", "hyde"]

    all_results = []

    for chunking in chunking_strategies:
        for retrieval in retrieval_methods:
            try:
                result = run_single_experiment(
                    chunking=chunking,
                    retrieval=retrieval,
                    eval_file=eval_file,
                    no_wandb=no_wandb,
                    llm_provider=llm_provider,
                )
                all_results.append(result)
            except Exception as e:
                print(f"\n[ERROR] {chunking}_{retrieval} failed: {e}")
                all_results.append({
                    "experiment": f"{chunking}_{retrieval}",
                    "chunking": chunking,
                    "retrieval": retrieval,
                    "error": str(e),
                })

    return all_results


def print_comparison_table(results_dir: Path = RESULTS_DIR) -> None:
    """Print a formatted comparison table from saved experiment results."""
    if not results_dir.exists():
        print(f"[ERROR] No results found in {results_dir}")
        print("        Run experiments first with --all")
        return

    results = []
    for f in sorted(results_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if "error" not in data:
            results.append(data)

    if not results:
        print("No successful experiment results found.")
        return

    # Print table
    print(f"\n{'='*85}")
    print(f"  EXPERIMENT COMPARISON TABLE")
    print(f"  (Copy this into your README and blog post)")
    print(f"{'='*85}")
    print(f"\n  {'Experiment':<28} {'Chunks':>7} {'Hit@3':>7} {'Hit@5':>7} {'Hit@10':>7} {'MRR':>7}")
    print(f"  {'-'*28} {'-'*7} {'-'*7} {'-'*7} {'-'*7} {'-'*7}")

    for r in results:
        name = r["experiment"]
        print(
            f"  {name:<28} "
            f"{r['num_chunks']:>7} "
            f"{r['hit_rate_3']:>7.3f} "
            f"{r['hit_rate_5']:>7.3f} "
            f"{r['hit_rate_10']:>7.3f} "
            f"{r['mrr']:>7.3f}"
        )

    # Find best configuration
    best = max(results, key=lambda r: r.get("hit_rate_5", 0))
    baseline = next((r for r in results if r["experiment"] == "fixed_512_naive"), None)

    print(f"\n  Best config: {best['experiment']}")
    print(f"  Best Hit@5:  {best['hit_rate_5']:.3f}")

    if baseline and best["experiment"] != "fixed_512_naive":
        improvement = best["hit_rate_5"] - baseline["hit_rate_5"]
        pct = (improvement / max(baseline["hit_rate_5"], 0.001)) * 100
        print(f"  vs Baseline:  +{improvement:.3f} ({pct:+.1f}%)")

    # Print markdown table for README
    print(f"\n\n  MARKDOWN TABLE (paste into README.md):")
    print(f"  ```")
    print(f"  | Configuration | Chunks | Hit@3 | Hit@5 | Hit@10 | MRR |")
    print(f"  |---|---|---|---|---|---|")
    for r in results:
        print(
            f"  | {r['experiment']} | {r['num_chunks']} | "
            f"{r['hit_rate_3']:.3f} | {r['hit_rate_5']:.3f} | "
            f"{r['hit_rate_10']:.3f} | {r['mrr']:.3f} |"
        )
    print(f"  ```")


def main():
    parser = argparse.ArgumentParser(description="Day 6: Run experiment matrix")
    parser.add_argument("--all", action="store_true", help="Run full experiment matrix")
    parser.add_argument("--compare", action="store_true", help="Print comparison table")
    parser.add_argument("--chunking", type=str, choices=list(CHUNKING_STRATEGIES.keys()))
    parser.add_argument("--retrieval", type=str, choices=["naive", "hyde"])
    parser.add_argument("--eval_file", type=Path, default=Path("evaluation/gold_qa.json"))
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--llm_provider",
        type=str,
        choices=["anthropic", "openai"],
        default="anthropic",
        help="LLM provider for HyDE hypothesis generation",
    )
    args = parser.parse_args()

    if args.compare:
        print_comparison_table()
    elif args.all:
        results = run_all_experiments(
            eval_file=args.eval_file,
            no_wandb=args.no_wandb,
            llm_provider=args.llm_provider,
        )
        print_comparison_table()
    elif args.chunking and args.retrieval:
        run_single_experiment(
            chunking=args.chunking,
            retrieval=args.retrieval,
            eval_file=args.eval_file,
            no_wandb=args.no_wandb,
            llm_provider=args.llm_provider,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
