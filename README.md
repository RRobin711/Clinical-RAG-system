# Clinical RAG System

Domain-specific RAG system for querying clinical psychology literature (MADRS, PHQ-9, ADHD-RS scales).

## Project Structure
```
clinical-rag-system/
├── data/
│   ├── papers/          # Raw PDFs
│   ├── extracted/       # Extracted text
│   └── chunks/          # Chunked documents
├── src/                 # Source code
├── tests/               # Test suite
└── notebooks/           # Analysis notebooks
```

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Status
- [x] Day 0-1: Project setup
- [ ] Day 2: Document ingestion
- [ ] Day 3: Embeddings + vector store
- [ ] Day 4: LLM integration
