from __future__ import annotations

import argparse
import hashlib
import signal
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from clinical_rag.io_utils import ensure_dir, append_jsonl, write_text
from clinical_rag.pdf_extract import extract_text_from_pdf


def stable_doc_id(pdf_path: Path) -> str:
    h = hashlib.sha1()
    h.update(pdf_path.name.encode("utf-8"))
    try:
        h.update(str(pdf_path.stat().st_size).encode("utf-8"))
    except FileNotFoundError:
        pass
    return h.hexdigest()[:12]


class TimeoutError(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutError("PDF extraction timed out")


def ingest_pdfs(pdf_dir: Path, out_dir: Path, manifest_path: Path) -> None:
    ensure_dir(out_dir)
    
    pdfs = sorted([p for p in pdf_dir.glob("*.pdf") if p.is_file()])
    if not pdfs:
        raise RuntimeError(f"No PDFs found in {pdf_dir}. Put PDFs into that folder first.")
    
    rows: List[Dict] = []
    
    for pdf_path in tqdm(pdfs, desc="Ingesting PDFs", unit="pdf"):
        doc_id = stable_doc_id(pdf_path)
        
        # Set timeout for 30 seconds per PDF
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            full_text, pages = extract_text_from_pdf(pdf_path)
            signal.alarm(0)  # Cancel the alarm
        except TimeoutError:
            signal.alarm(0)
            rows.append(
                {
                    "doc_id": doc_id,
                    "source_pdf": pdf_path.name,
                    "status": "timeout",
                    "error": "PDF extraction timed out after 30 seconds",
                }
            )
            print(f"\n⚠️  TIMEOUT: {pdf_path.name} - skipping")
            continue
        except Exception as e:
            signal.alarm(0)
            rows.append(
                {
                    "doc_id": doc_id,
                    "source_pdf": pdf_path.name,
                    "status": "error",
                    "error": repr(e),
                }
            )
            print(f"\n⚠️  ERROR: {pdf_path.name} - {repr(e)}")
            continue
        
        if len(full_text) < 500:
            rows.append(
                {
                    "doc_id": doc_id,
                    "source_pdf": pdf_path.name,
                    "status": "skipped_low_text",
                    "num_chars": len(full_text),
                    "num_pages": len(pages),
                }
            )
            continue
        
        out_path = out_dir / f"{doc_id}.txt"
        write_text(out_path, full_text)
        
        rows.append(
            {
                "doc_id": doc_id,
                "source_pdf": pdf_path.name,
                "status": "ok",
                "num_chars": len(full_text),
                "num_pages": len(pages),
                "extracted_txt": str(out_path.as_posix()),
            }
        )
    
    append_jsonl(manifest_path, rows)
    print(f"\nDone. Wrote extracted texts to: {out_dir}")
    print(f"Manifest appended to: {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Day 2: Extract text from PDFs using PyMuPDF.")
    parser.add_argument("--pdf_dir", type=Path, default=Path("data/papers"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/extracted"))
    parser.add_argument("--manifest", type=Path, default=Path("data/extraction_manifest.jsonl"))
    args = parser.parse_args()
    
    ingest_pdfs(args.pdf_dir, args.out_dir, args.manifest)


if __name__ == "__main__":
    main()