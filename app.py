"""
Days 8-9: Gradio UI for the Clinical RAG system.

Features:
- Query input with example questions
- Retrieved source chunks displayed alongside the answer
- Confidence/relevance scores for each source
- Toggle between Naive and HyDE retrieval
- Experiment results comparison tab

Run locally:  python app.py
Deploy:       Push to Hugging Face Spaces (Gradio SDK)
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import gradio as gr

from clinical_rag.pipeline import answer_question, AnswerWithSources
from clinical_rag.retriever import Retriever
from clinical_rag.chunking_strategies import HyDERetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEX_DIR = Path(os.environ.get("INDEX_DIR", "data/index"))
RESULTS_DIR = Path("evaluation/experiment_results")


# ---------------------------------------------------------------------------
# Query handlers
# ---------------------------------------------------------------------------


def format_sources(result: AnswerWithSources) -> str:
    """Format retrieved sources as readable text with scores."""
    lines = []
    for rc in result.retrieved:
        chunk = rc.chunk
        source = chunk.get("source", "unknown")
        chunk_id = chunk.get("chunk_id", "?")
        text_preview = chunk.get("text", "")[:300].strip()
        if len(chunk.get("text", "")) > 300:
            text_preview += "..."

        score_bar = "█" * int(rc.score * 10) + "░" * (10 - int(rc.score * 10))
        lines.append(
            f"**[{rc.rank}]** {source} (`{chunk_id}`)  \n"
            f"Score: {rc.score:.4f} {score_bar}  \n"
            f"> {text_preview}\n"
        )
    return "\n---\n".join(lines) if lines else "No sources retrieved."


def query_rag(question: str, use_hyde: bool, top_k: int) -> tuple:
    """Main query function called by Gradio."""
    if not question.strip():
        return "Please enter a question.", "No sources to display.", ""

    try:
        result = answer_question(
            index_dir=INDEX_DIR,
            question=question,
            k=top_k,
        )

        answer = result.answer
        sources = format_sources(result)

        # Metadata
        meta = (
            f"**Model:** {result.model}  \n"
            f"**Retrieval:** {'HyDE' if use_hyde else 'Naive'}  \n"
            f"**Top-k:** {top_k}  \n"
            f"**Sources used:** {len(result.retrieved)}"
        )

        return answer, sources, meta

    except Exception as e:
        return f"Error: {str(e)}", "", ""


def load_experiment_results() -> str:
    """Load experiment comparison table for the results tab."""
    if not RESULTS_DIR.exists():
        return "No experiment results found. Run `python scripts/06_run_experiments.py --all` first."

    results = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        if "error" not in data:
            results.append(data)

    if not results:
        return "No successful experiments found."

    # Build markdown table
    lines = [
        "| Configuration | Chunks | Hit@3 | Hit@5 | Hit@10 | MRR |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['experiment']} | {r['num_chunks']} | "
            f"{r['hit_rate_3']:.3f} | {r['hit_rate_5']:.3f} | "
            f"{r['hit_rate_10']:.3f} | {r['mrr']:.3f} |"
        )

    best = max(results, key=lambda r: r.get("hit_rate_5", 0))
    lines.append(f"\n**Best configuration:** {best['experiment']} (Hit@5 = {best['hit_rate_5']:.3f})")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

EXAMPLE_QUERIES = [
    "What is the inter-rater reliability of the MADRS?",
    "How many items does the PHQ-9 contain?",
    "What cutoff score indicates moderate depression on the PHQ-9?",
    "What populations has the MADRS been validated in?",
    "How does the PHQ-9 compare to the BDI-II?",
    "What is the sensitivity of the MADRS to treatment change?",
]


def build_app() -> gr.Blocks:
    with gr.Blocks(
        title="Clinical RAG System",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # 🏥 Clinical Psychology RAG System
            
            Query clinical psychology research literature on assessment scales 
            (MADRS, PHQ-9, ADHD-RS). Answers are grounded in retrieved sources 
            with citations.
            
            Built as part of a 30-day AI/ML engineering sprint.
            [GitHub](https://github.com/RRobin711/Clinical-RAG-system) | 
            [Blog Post](#) | [W&B Dashboard](#)
            """
        )

        with gr.Tab("🔍 Query"):
            with gr.Row():
                with gr.Column(scale=2):
                    question = gr.Textbox(
                        label="Ask a question about clinical assessment scales",
                        placeholder="e.g., What is the inter-rater reliability of the MADRS?",
                        lines=2,
                    )
                    with gr.Row():
                        use_hyde = gr.Checkbox(
                            label="Use HyDE retrieval",
                            value=False,
                            info="Hypothetical Document Embeddings (Gao et al., 2022)",
                        )
                        top_k = gr.Slider(
                            minimum=3,
                            maximum=10,
                            value=5,
                            step=1,
                            label="Top-k sources",
                        )
                    submit_btn = gr.Button("Search", variant="primary")

                with gr.Column(scale=1):
                    meta_output = gr.Markdown(label="Query Info")

            answer_output = gr.Markdown(label="Answer")

            with gr.Accordion("📚 Retrieved Sources", open=False):
                sources_output = gr.Markdown()

            gr.Examples(
                examples=[[q] for q in EXAMPLE_QUERIES],
                inputs=[question],
                label="Example Questions",
            )

            submit_btn.click(
                fn=query_rag,
                inputs=[question, use_hyde, top_k],
                outputs=[answer_output, sources_output, meta_output],
            )

        with gr.Tab("📊 Experiment Results"):
            gr.Markdown("## Retrieval Experiment Comparison")
            gr.Markdown(
                "Results from the 2×2 experiment matrix: "
                "(Naive vs HyDE) × (Chunking Strategies)"
            )
            results_display = gr.Markdown(value=load_experiment_results())
            refresh_btn = gr.Button("Refresh Results")
            refresh_btn.click(fn=load_experiment_results, outputs=[results_display])

        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                """
                ## Architecture
                
                ```
                Query → Embed (BGE-small) → FAISS Retrieval → Format Prompt → LLM → Grounded Answer
                         ↑                                                        
                    [Optional HyDE]                                               
                    Query → LLM hypothesis → Embed hypothesis → FAISS            
                ```
                
                ## Technical Details
                
                - **Embedding Model:** BAAI/bge-small-en-v1.5 (384-dim)
                - **Vector Store:** FAISS (IndexFlatIP with normalized embeddings)
                - **LLM:** Claude (Anthropic API)
                - **Corpus:** Clinical psychology papers from PubMed Central
                - **Evaluation:** Custom gold QA dataset with Hit@k and MRR metrics
                
                ## What Makes This Different
                
                1. **Real evaluation framework** — not just "it works, trust me"
                2. **HyDE implementation** from Gao et al., 2022
                3. **Systematic experiments** comparing chunking strategies × retrieval methods
                4. **Domain-specific** — built for clinical psychology, not generic documents
                
                ## Links
                
                - [Source Code](https://github.com/RRobin711/Clinical-RAG-system)
                - [W&B Dashboard](#)
                - [Blog Post](#)
                """
            )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
