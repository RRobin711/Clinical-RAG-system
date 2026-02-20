#!/usr/bin/env python3
"""
Day 5: Build gold-standard evaluation dataset.

Two modes:
  --list_docs     Print your doc_ids so you know what to reference
  --template      Generate a starter template with example questions
  --validate      Check that all doc_ids in your gold QA actually exist

Usage:
    python scripts/05_build_eval_dataset.py --list_docs
    python scripts/05_build_eval_dataset.py --template --out evaluation/gold_qa.json
    python scripts/05_build_eval_dataset.py --validate --eval_file evaluation/gold_qa.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def list_docs(chunks_file: Path) -> None:
    """Print all unique doc_ids and their source PDFs."""
    if not chunks_file.exists():
        print(f"[ERROR] Chunks file not found: {chunks_file}")
        print("        Run your chunking script first.")
        return

    doc_map = {}  # doc_id -> {source, num_chunks, total_chars}
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            doc_id = row.get("doc_id", "unknown")
            if doc_id not in doc_map:
                doc_map[doc_id] = {
                    "source": row.get("source", "unknown"),
                    "num_chunks": 0,
                    "total_chars": 0,
                }
            doc_map[doc_id]["num_chunks"] += 1
            doc_map[doc_id]["total_chars"] += row.get("num_chars", 0)

    print(f"\n{'doc_id':<15} {'chunks':>7} {'chars':>8}   source_pdf")
    print("-" * 80)
    for doc_id, info in sorted(doc_map.items()):
        print(
            f"{doc_id:<15} {info['num_chunks']:>7} {info['total_chars']:>8}   {info['source']}"
        )
    print(f"\nTotal: {len(doc_map)} documents, "
          f"{sum(d['num_chunks'] for d in doc_map.values())} chunks")
    print(f"\nCopy the doc_id values above into your gold QA dataset.")


def generate_template(out_path: Path) -> None:
    """Generate a starter template with example questions to fill in."""
    template = [
        {
            "question": "What is the inter-rater reliability of the MADRS?",
            "answer": "REPLACE: e.g., ICC = 0.89 (95% CI: 0.82-0.94) as reported by Smith et al.",
            "source_doc_id": "REPLACE_WITH_YOUR_DOC_ID",
            "difficulty": "medium",
            "category": "reliability",
            "notes": "Look for ICC, kappa, or correlation coefficients",
        },
        {
            "question": "How many items does the PHQ-9 contain?",
            "answer": "REPLACE: e.g., The PHQ-9 contains 9 items scored 0-3, with total scores ranging from 0-27.",
            "source_doc_id": "REPLACE_WITH_YOUR_DOC_ID",
            "difficulty": "easy",
            "category": "scoring",
            "notes": "Basic factual question - should be easy to retrieve",
        },
        {
            "question": "What cutoff score on the PHQ-9 indicates moderate depression?",
            "answer": "REPLACE: e.g., A score of 10-14 on the PHQ-9 indicates moderate depression severity.",
            "source_doc_id": "REPLACE_WITH_YOUR_DOC_ID",
            "difficulty": "easy",
            "category": "scoring",
            "notes": "",
        },
        {
            "question": "What is the sensitivity of MADRS for detecting treatment response?",
            "answer": "REPLACE with actual value from your papers",
            "source_doc_id": "REPLACE_WITH_YOUR_DOC_ID",
            "difficulty": "hard",
            "category": "validity",
            "notes": "Sensitivity to change is a specific psychometric property",
        },
        {
            "question": "What populations has the PHQ-9 been validated in?",
            "answer": "REPLACE: e.g., primary care, adolescents, geriatric patients, etc.",
            "source_doc_id": "REPLACE_WITH_YOUR_DOC_ID",
            "difficulty": "medium",
            "category": "population",
            "notes": "May span multiple papers - pick the one with broadest coverage",
        },
        {
            "question": "What is the internal consistency (Cronbach's alpha) of the MADRS?",
            "answer": "REPLACE with actual alpha value",
            "source_doc_id": "REPLACE_WITH_YOUR_DOC_ID",
            "difficulty": "medium",
            "category": "reliability",
            "notes": "",
        },
        {
            "question": "How does the PHQ-9 compare to the BDI-II for depression screening?",
            "answer": "REPLACE with comparative data from your papers",
            "source_doc_id": "REPLACE_WITH_YOUR_DOC_ID",
            "difficulty": "hard",
            "category": "validity",
            "notes": "Comparative questions are harder for retrieval",
        },
        {
            "question": "What are the subscales or factor structure of the MADRS?",
            "answer": "REPLACE with factor analysis results",
            "source_doc_id": "REPLACE_WITH_YOUR_DOC_ID",
            "difficulty": "hard",
            "category": "structure",
            "notes": "Factor structure questions test deeper retrieval",
        },
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[OK] Template written to: {out_path}")
    print(f"     Contains {len(template)} example questions.")
    print(f"\n     NEXT STEPS:")
    print(f"     1. Run: python scripts/05_build_eval_dataset.py --list_docs")
    print(f"     2. Open {out_path} in your editor")
    print(f"     3. Replace EVERY 'REPLACE_WITH_YOUR_DOC_ID' with real doc_ids")
    print(f"     4. Replace EVERY 'REPLACE:' answer with actual answers from your papers")
    print(f"     5. Add 22+ more questions (target: 30-50 total)")
    print(f"     6. Mix categories: reliability, validity, scoring, population, structure")


def validate_eval_dataset(eval_file: Path, chunks_file: Path) -> None:
    """Validate that all doc_ids in the eval dataset exist in chunks."""
    if not eval_file.exists():
        print(f"[ERROR] Eval file not found: {eval_file}")
        return
    if not chunks_file.exists():
        print(f"[ERROR] Chunks file not found: {chunks_file}")
        return

    # Get all doc_ids from chunks
    known_doc_ids = set()
    with open(chunks_file, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            known_doc_ids.add(row.get("doc_id", ""))

    # Check eval dataset
    items = json.loads(eval_file.read_text(encoding="utf-8"))
    missing = []
    placeholder = []
    categories = Counter()

    for i, item in enumerate(items):
        doc_id = item.get("source_doc_id", "")
        if "REPLACE" in doc_id:
            placeholder.append(i)
        elif doc_id not in known_doc_ids:
            missing.append((i, doc_id))
        categories[item.get("category", "unknown")] += 1

    print(f"\n{'='*60}")
    print(f"  Eval Dataset Validation: {eval_file.name}")
    print(f"{'='*60}")
    print(f"  Total questions:    {len(items)}")
    print(f"  Known doc_ids:      {len(known_doc_ids)}")
    print(f"  Placeholder entries: {len(placeholder)}")
    print(f"  Missing doc_ids:    {len(missing)}")

    if placeholder:
        print(f"\n  ⚠  Questions still have REPLACE placeholders: indices {placeholder}")

    if missing:
        print(f"\n  ✗  Questions reference non-existent doc_ids:")
        for idx, did in missing:
            print(f"     [{idx}] doc_id='{did}'")

    valid = len(items) - len(placeholder) - len(missing)
    print(f"\n  ✓  Valid questions: {valid}/{len(items)}")

    print(f"\n  Category distribution:")
    for cat, count in categories.most_common():
        print(f"     {cat}: {count}")

    if valid >= 30:
        print(f"\n  ✓  You have enough questions for meaningful evaluation!")
    else:
        print(f"\n  ⚠  Target 30+ questions. You have {valid} valid ones.")


def main():
    parser = argparse.ArgumentParser(description="Day 5: Build gold evaluation dataset")
    parser.add_argument("--list_docs", action="store_true", help="List all doc_ids from chunks")
    parser.add_argument("--template", action="store_true", help="Generate starter QA template")
    parser.add_argument("--validate", action="store_true", help="Validate eval dataset")
    parser.add_argument("--chunks_file", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--eval_file", type=Path, default=Path("evaluation/gold_qa.json"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.list_docs:
        list_docs(args.chunks_file)
    elif args.template:
        out = args.out or args.eval_file
        generate_template(out)
    elif args.validate:
        validate_eval_dataset(args.eval_file, args.chunks_file)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
