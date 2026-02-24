# What Broke When I Built a Clinical RAG System (And How I Fixed It)

*A practical guide to building, evaluating, and improving a domain-specific RAG system for clinical psychology literature.*

---

## Why Clinical Psychology? Why RAG?

*(2-3 paragraphs: Your story. Connect to Sama Therapeutics work. Why this domain, why this approach.)*

I work on clinical AI systems at Sama Therapeutics, building tools that help clinicians assess conditions like depression using standardized scales — MADRS, PHQ-9, ADHD-RS. These scales have decades of validation research behind them, scattered across thousands of papers.

The problem: when a clinician asks "What's the inter-rater reliability of the MADRS in geriatric populations?", the answer exists in the literature, but finding it means searching PubMed, reading abstracts, and hoping you find the right paper. A RAG system can make that knowledge instantly queryable.

Here's what I built, what broke, and what I learned fixing it.

---

## The Architecture (Keep It Simple)

*(Include your architecture diagram. Explain the pipeline in 3-4 sentences.)*

```
PDFs → Extract → Chunk → Embed (BGE-small) → FAISS Index
Query → Embed → Retrieve top-k → Format prompt → Claude → Cited answer
```

The pipeline is intentionally straightforward: no LangChain abstraction layers, no complex orchestration. Every component is a Python function I can test and debug independently.

---

## Failure #1: Chunking Destroyed Clinical Context

*(This is your strongest section. Show real examples.)*

My first chunking strategy was naive: split every 512 characters with 50-character overlap. This worked for short passages but destroyed the structure of clinical papers.

**The problem in practice:** A sentence like "The ICC for the MADRS was 0.89" would end up in one chunk, while the critical context — "in a sample of 200 primary care patients with moderate depression" — landed in the next chunk. The retriever would find the number but lose the population context, and the LLM would report reliability without qualification.

**What I tried:**
- Fixed 512-char chunks (baseline): Hit@5 = X.XX
- Fixed 1024-char chunks: Hit@5 = X.XX (bigger context, less precision)
- Semantic chunking (section-boundary aware): Hit@5 = X.XX

*(Insert your actual metrics table here. Even if the numbers are modest, the fact that you measured them is the point.)*

**What I learned:** Clinical papers have highly structured information — Abstract, Methods, Results, Discussion. Chunking that respects these boundaries preserves the logical units that clinicians actually query.

---

## Failure #2: Questions and Answers Don't Live in the Same Embedding Space

*(This is the HyDE section. This is what makes your project stand out.)*

My retrieval was mediocre even with good chunks. The reason was subtle: when someone asks "What is the inter-rater reliability of MADRS?", the embedding of that question is linguistically very different from the embedding of "The ICC was 0.89 (95% CI: 0.82–0.94)."

Questions are interrogative. Answers are declarative. Dense retrieval assumes they'll be close in embedding space. They often aren't.

**The fix: HyDE (Hypothetical Document Embeddings)**

From Gao et al., 2022: instead of embedding the question, first ask the LLM to generate a hypothetical expert answer, then embed *that*. The hypothetical answer is linguistically closer to the real answer chunks in the corpus.

```python
# The core idea in ~10 lines
hypothesis = llm("Write a factual paragraph answering: " + question)
hypothesis_embedding = embed(hypothesis)  # NOT the question
retrieved_chunks = faiss_index.search(hypothesis_embedding, k=5)
final_answer = llm(question + retrieved_chunks)  # Still grounded in real sources
```

**Results:**
| Config | Hit@5 | MRR |
|---|---|---|
| Naive + fixed_512 | X.XX | X.XX |
| HyDE + semantic | X.XX | X.XX |

*(Insert your real 2×2 comparison table.)*

**Where HyDE fails:** It adds latency (one extra LLM call per query), costs more, and can hallucinate a wrong hypothesis that pulls retrieval in the wrong direction. For clinical applications where accuracy matters more than speed, the tradeoff is worth it. For real-time chat, it probably isn't.

---

## Failure #3: The System Hallucinated Confidently from Bad Context

*(Show an example of the LLM answering from wrong chunks.)*

Without evaluation, I had no idea my system was doing this. It would retrieve a chunk about PHQ-9 when asked about MADRS (because both papers discuss "depression" and "reliability"), and Claude would confidently synthesize an answer from the wrong scale's data.

**Building the evaluation dataset** was the single most important thing I did. I wrote 30+ question-answer pairs with known source documents, then measured whether the retriever was even finding the right chunks before the LLM touched them.

*(Show a W&B screenshot or chart here.)*

---

## What I'd Do Differently With More Time

1. **Reranking.** A cross-encoder reranker after FAISS retrieval would catch cases where the bi-encoder embedding misses nuance.
2. **Multi-hop retrieval.** Some questions require synthesizing across multiple papers — the current system retrieves from one document at a time.
3. **Fine-tuned embeddings.** BGE-small is general-purpose. A model fine-tuned on clinical text (e.g., PubMedBERT embeddings) might close the domain gap.
4. **Real clinician feedback.** My gold QA dataset was self-authored. Clinician-written questions would test retrieval on patterns I can't anticipate.

---

## Try It Yourself

- **Live demo:** [Hugging Face Spaces link]
- **Source code:** [GitHub link]
- **W&B experiments:** [Dashboard link]

---

*Tags: Machine Learning, RAG, NLP, AI, Healthcare, Clinical Psychology*

*Word count target: 1,800–2,500 words. The HyDE section alone justifies the length.*
