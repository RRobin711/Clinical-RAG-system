from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def ensure_dir(path: Path) -> None:
    """Ensure a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)


def read_text(path: Path) -> str:
    """Read text from a file."""
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    """Write text to a file."""
    path.write_text(text, encoding="utf-8")


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """Read JSONL file line by line."""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Write rows to a JSONL file."""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Append rows to a JSONL file."""
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")