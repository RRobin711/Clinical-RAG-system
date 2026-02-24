"""Pytest configuration - ensure src/clinical_rag is importable."""
import sys
from pathlib import Path

# Add src/ to path so tests can import clinical_rag
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
