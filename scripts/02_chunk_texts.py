from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from clinical_rag.chunking import chunk_text, estimate_tokens
from clinical_rag.io_utils import read_jsonl, write_jsonl, read_text


def load_manifest_map(manifest_path: Path) -> Dict[str, Dict]:
    if not manifest_path.exists():
        return {}
    
    m: Dict[str, Dict] = {}
    for row in read_jsonl(manifest_path):
        if row.get("status") == "ok" and row.get("doc_id"):
            m[row["doc_id"]] = row
    return m


def chunk_all(
    in_dir: Path, 
    out_file: Path, 
    manifest_path: Path,
    chunk_size_chars: int, 
    overlap_chars: int
) -> None:
    manifest = load_manifest_map(manifest_path)
    
    txt_files = sorted([p for p in in_dir.glob("*.txt") if p.is_file()])
    if not txt_files:
        raise RuntimeError(f"No extracted .txt files found in {in_dir}. Run scripts/01_ingest_pdfs.py first.")
    
    rows: List[Dict] = []
    
    for txt_path in tqdm(txt_files, desc="Chunking texts", unit="doc"):
        doc_id = txt_path.stem
        text = read_text(txt_path)
        
        chunks = chunk_text(
            text, 
            chunk_size_chars=chunk_size_chars,
            overlap_chars=overlap_chars
        )
        
        source_pdf = manifest.get(doc_id, {}).get("source_pdf", f"{doc_id}.pdf")
        
        for i, ch in enumerate(chunks):
            rows.append(
                {
                    "chunk_id": f"{doc_id}_{i:05d}",
                    "doc_id": doc_id,
                    "source": source_pdf,
                    "chunk_index": i,
                    "num_chars": len(ch),
                    "token_estimate": estimate_tokens(ch),
                    "text": ch,
                }
            )
    
    write_jsonl(out_file, rows)
    print(f"Done. Wrote {len(rows)} chunks to: {out_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 2: Chunk extracted texts into overlapping windows.")
    parser.add_argument("--in_dir", type=Path, default=Path("data/extracted"))
    parser.add_argument("--out_file", type=Path, default=Path("data/chunks.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/extraction_manifest.jsonl"))
    parser.add_argument("--chunk_size_chars", type=int, default=2000)
    parser.add_argument("--overlap_chars", type=int, default=200)
    args = parser.parse_args()
    
    chunk_all(
        args.in_dir,
        args.out_file,
        args.manifest,
        chunk_size_chars=args.chunk_size_chars,
        overlap_chars=args.overlap_chars,
    )


if __name__ == "__main__":
    main()