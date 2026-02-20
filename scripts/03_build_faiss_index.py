from __future__ import annotations

import argparse
from pathlib import Path

from clinical_rag.indexing import build_faiss_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 3: Embed chunks and build a FAISS index.")
    parser.add_argument("--chunks_file", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--index_dir", type=Path, default=Path("data/index"))
    parser.add_argument("--embedding_model", type=str, default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--metric", choices=["ip", "l2"], default="ip")
    parser.add_argument("--normalize", action="store_true", help="L2-normalize embeddings (recommended for cosine-like search).")
    parser.add_argument("--no_normalize", action="store_true", help="Disable normalization.")
    parser.add_argument(
        "--query_instruction",
        type=str,
        default=None,
        help="Optional prefix added to queries before embedding (not added to corpus). "
             "If omitted, a good default is used for some models (e.g., BAAI/bge-small-en-v1.5). "
             "Set to empty string to disable.",
    )
    args = parser.parse_args()
    
    normalize = True
    if args.no_normalize:
        normalize = False
    if args.normalize:
        normalize = True
    
    artifacts = build_faiss_index(
        chunks_file=args.chunks_file,
        index_dir=args.index_dir,
        embedding_model_name=args.embedding_model,
        batch_size=args.batch_size,
        normalize_embeddings=normalize,
        metric=args.metric,
        query_instruction=args.query_instruction,
    )
    
    print("Built index artifacts:")
    print(f"- {artifacts.index_path}")
    print(f"- {artifacts.chunks_path}")
    print(f"- {artifacts.meta_path}")


if __name__ == "__main__":
    main()