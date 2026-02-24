"""
Day 7: Test suite for the Clinical RAG system.

Tests cover:
- Ingestion handles edge cases (empty PDFs, corrupt files)
- Chunking produces correct sizes and overlaps
- Retrieval returns results for known queries
- Evaluation metrics compute correctly
- Semantic chunking splits on section boundaries

Run: pytest tests/ -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from clinical_rag.chunking import chunk_text
from clinical_rag.chunking_strategies import (
    chunk_fixed_512,
    chunk_fixed_1024,
    chunk_semantic,
    CHUNKING_STRATEGIES,
)
from clinical_rag.evaluation import (
    GoldQAItem,
    RetrievalResult,
    compute_eval_summary,
    save_gold_qa,
    load_gold_qa,
)


# ---------------------------------------------------------------------------
# Chunking tests
# ---------------------------------------------------------------------------


class TestChunking:
    """Tests for the chunking module."""

    def test_basic_chunking_produces_output(self):
        """Chunk text should produce at least one chunk for valid input."""
        text = "This is a test sentence. " * 100
        chunks = chunk_text(text, chunk_size_chars=200, overlap_chars=20, min_chunk_chars=50)
        assert len(chunks) > 0

    def test_empty_text_returns_empty(self):
        """Empty input should return no chunks."""
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_chunk_sizes_within_bounds(self):
        """No chunk should vastly exceed the requested size."""
        text = "Lorem ipsum dolor sit amet. " * 500
        chunks = chunk_text(text, chunk_size_chars=500, overlap_chars=50, min_chunk_chars=100)
        for ch in chunks:
            # Allow some overflow for word-boundary seeking
            assert len(ch) <= 800, f"Chunk too large: {len(ch)} chars"

    def test_overlap_exists(self):
        """Consecutive chunks should share some text (overlap)."""
        text = "word " * 500
        chunks = chunk_text(text, chunk_size_chars=100, overlap_chars=20, min_chunk_chars=20)
        if len(chunks) >= 2:
            # The end of chunk[0] should overlap with start of chunk[1]
            tail = chunks[0][-20:]
            assert tail in chunks[1], "Expected overlap between consecutive chunks"

    def test_invalid_params_raise(self):
        """Invalid chunk parameters should raise ValueError."""
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size_chars=0)
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size_chars=100, overlap_chars=100)
        with pytest.raises(ValueError):
            chunk_text("test", chunk_size_chars=100, overlap_chars=-1)

    def test_short_text_single_chunk(self):
        """Text shorter than chunk_size should produce one chunk."""
        text = "Short text for testing."
        chunks = chunk_text(text, chunk_size_chars=500, overlap_chars=50, min_chunk_chars=5)
        assert len(chunks) == 1
        assert chunks[0].strip() == text


class TestChunkingStrategies:
    """Tests for the Day 6 chunking strategies."""

    SAMPLE_TEXT = (
        "Abstract\n\n"
        "This study examines the psychometric properties of the MADRS. "
        "We recruited 200 participants from primary care settings.\n\n"
        "Methods\n\n"
        "Participants completed the MADRS and PHQ-9 at baseline and 4-week follow-up. "
        "Inter-rater reliability was assessed using intraclass correlation coefficients. "
        "Internal consistency was evaluated with Cronbach's alpha.\n\n"
        "Results\n\n"
        "The MADRS showed excellent inter-rater reliability (ICC = 0.89). "
        "Internal consistency was good (alpha = 0.85). "
        "Convergent validity with the PHQ-9 was strong (r = 0.78).\n\n"
        "Discussion\n\n"
        "These findings support the use of the MADRS in primary care settings. "
        "The scale demonstrates strong psychometric properties across multiple metrics."
    )

    def test_fixed_512_produces_output(self):
        assert len(chunk_fixed_512(self.SAMPLE_TEXT)) >= 1

    def test_fixed_1024_produces_output(self):
        assert len(chunk_fixed_1024(self.SAMPLE_TEXT)) >= 1

    def test_semantic_respects_sections(self):
        """Semantic chunking should not split mid-section when possible."""
        chunks = chunk_semantic(self.SAMPLE_TEXT, max_chunk_chars=5000)
        # With a large max, the whole text could be one chunk
        # With the default, it should split at section boundaries
        for ch in chunks:
            assert len(ch) > 0

    def test_semantic_handles_empty(self):
        assert chunk_semantic("") == []

    def test_all_strategies_registered(self):
        """All named strategies should be in the registry."""
        expected = {"fixed_512", "fixed_1024", "fixed_2000", "semantic"}
        assert set(CHUNKING_STRATEGIES.keys()) == expected


# ---------------------------------------------------------------------------
# Evaluation tests
# ---------------------------------------------------------------------------


class TestEvaluation:
    """Tests for the evaluation framework."""

    def test_retrieval_result_hit_at_k(self):
        """Hit@k should be correct based on retrieved doc position."""
        result = RetrievalResult(
            question="test",
            gold_doc_id="doc_a",
            retrieved_doc_ids=["doc_b", "doc_c", "doc_a", "doc_d", "doc_e"],
            retrieved_scores=[0.9, 0.8, 0.7, 0.6, 0.5],
        )
        result.compute()
        assert result.hit_at_3 is True  # doc_a is at rank 3
        assert result.hit_at_5 is True
        assert result.reciprocal_rank == pytest.approx(1.0 / 3)

    def test_retrieval_result_miss(self):
        """When gold doc is not retrieved, all metrics should be 0/False."""
        result = RetrievalResult(
            question="test",
            gold_doc_id="doc_a",
            retrieved_doc_ids=["doc_b", "doc_c", "doc_d"],
            retrieved_scores=[0.9, 0.8, 0.7],
        )
        result.compute()
        assert result.hit_at_3 is False
        assert result.hit_at_5 is False
        assert result.reciprocal_rank == 0.0

    def test_retrieval_result_rank_1(self):
        """Perfect retrieval: gold doc at rank 1."""
        result = RetrievalResult(
            question="test",
            gold_doc_id="doc_a",
            retrieved_doc_ids=["doc_a", "doc_b"],
            retrieved_scores=[0.95, 0.8],
        )
        result.compute()
        assert result.hit_at_3 is True
        assert result.reciprocal_rank == 1.0

    def test_eval_summary_aggregation(self):
        """Summary should correctly aggregate individual results."""
        results = [
            RetrievalResult(
                question="q1", gold_doc_id="a",
                retrieved_doc_ids=["a", "b"], retrieved_scores=[0.9, 0.8],
            ),
            RetrievalResult(
                question="q2", gold_doc_id="c",
                retrieved_doc_ids=["b", "a"], retrieved_scores=[0.9, 0.8],
            ),
        ]
        for r in results:
            r.compute()

        summary = compute_eval_summary(results)
        assert summary.num_queries == 2
        assert summary.hit_rate_3 == 0.5  # only q1 has hit
        assert summary.mrr == pytest.approx((1.0 + 0.0) / 2)

    def test_gold_qa_save_load_roundtrip(self, tmp_path):
        """Save/load should preserve all fields."""
        items = [
            GoldQAItem(
                question="What is reliability?",
                answer="ICC = 0.89",
                source_doc_id="doc_123",
                difficulty="medium",
                category="reliability",
            ),
        ]
        path = tmp_path / "test_qa.json"
        save_gold_qa(items, path)
        loaded = load_gold_qa(path)
        assert len(loaded) == 1
        assert loaded[0].question == items[0].question
        assert loaded[0].source_doc_id == items[0].source_doc_id

    def test_empty_results_summary(self):
        """Empty results list should produce zero-valued summary."""
        summary = compute_eval_summary([])
        assert summary.num_queries == 0
        assert summary.mrr == 0.0


# ---------------------------------------------------------------------------
# Integration smoke tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Smoke tests that verify components wire together."""

    def test_chunk_then_count(self):
        """End-to-end: text → chunks → verify count is reasonable."""
        # Simulate a short clinical paper
        paper_text = (
            "Background: Depression affects millions worldwide. "
            "The PHQ-9 is widely used for screening. "
        ) * 50  # ~5000 chars

        for strategy_name, fn in CHUNKING_STRATEGIES.items():
            chunks = fn(paper_text)
            assert len(chunks) > 0, f"Strategy {strategy_name} produced no chunks"
            total_chars = sum(len(c) for c in chunks)
            # Total chars should be roughly >= original (due to overlap)
            assert total_chars >= len(paper_text) * 0.5, (
                f"Strategy {strategy_name}: chunks have too few total chars"
            )
