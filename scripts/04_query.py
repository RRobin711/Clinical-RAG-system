from __future__ import annotations

import argparse
from pathlib import Path

from clinical_rag.pipeline import answer_question


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 4: End-to-end RAG query (retrieve top-k, then grounded answer).")
    parser.add_argument("--index_dir", type=Path, default=Path("data/index"))
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--show_sources", action="store_true")
    parser.add_argument("question", type=str, nargs="+")
    args = parser.parse_args()
    
    question = " ".join(args.question).strip()
    
    result = answer_question(index_dir=args.index_dir, question=question, k=args.k)
    
    print(f"\nModel: {result.model}\n")
    print(result.answer)
    print("\n---\nRetrieved sources:")
    for rc in result.retrieved:
        ch = rc.chunk
        print(f"{rc.rank}. {ch.get('chunk_id')} | score={rc.score:.4f} | {ch.get('source')}")
        if args.show_sources:
            print(ch.get("text", "")[:800].strip())
            print("")


if __name__ == "__main__":
    main()